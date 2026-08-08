package rewrite

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/kafo-play-box/shisui/internal/jsonl"
)

func TestMain(m *testing.M) {
	retryBackoff = []time.Duration{time.Millisecond, time.Millisecond}
	os.Exit(m.Run())
}

type mockChat struct {
	ts      *httptest.Server
	mu      sync.Mutex
	calls   int
	auth    string
	fail500 int
}

func newMockChat(t *testing.T) *mockChat {
	t.Helper()
	m := &mockChat{}
	mux := http.NewServeMux()
	mux.HandleFunc("/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		content := req.Messages[1].Content
		m.mu.Lock()
		m.calls++
		m.auth = r.Header.Get("Authorization")
		if strings.Contains(content, "fail500") {
			m.fail500++
		}
		n := m.calls
		m.mu.Unlock()
		switch {
		case strings.Contains(content, "fail500") && n < 3:
			w.WriteHeader(http.StatusInternalServerError)
			return
		case strings.Contains(content, "fail500"):
			fmt.Fprint(w, `{"choices":[{"message":{"content":"{\"simple_meaning\":\"recovered after retries\",\"quality\":\"medium\",\"selected_meaning\":\"to move fast\"}"}}]}`)
		case strings.Contains(content, "forbidden"):
			w.WriteHeader(http.StatusForbidden)
			return
		case strings.Contains(content, "xyzzy"):
			fmt.Fprint(w, `{"choices":[{"message":{"content":"not json"}}]}`)
		case strings.Contains(content, "paraphrase"):
			fmt.Fprint(w, `{"choices":[{"message":{"content":"{\"simple_meaning\":\"to move quickly on your feet\",\"quality\":\"good\",\"selected_meaning\":\"to go fast\"}"}}]}`)
		case strings.Contains(content, "no-selected"):
			fmt.Fprint(w, `{"choices":[{"message":{"content":"{\"simple_meaning\":\"to move quickly on your feet\",\"quality\":\"good\"}"}}]}`)
		case strings.Contains(content, "empty-selected"):
			fmt.Fprint(w, `{"choices":[{"message":{"content":"{\"simple_meaning\":\"to move quickly on your feet\",\"quality\":\"medium\",\"selected_meaning\":\"\"}"}}]}`)
		case strings.Contains(content, "weird"):
			fmt.Fprint(w, `{"choices":[{"message":{"content":"{\"simple_meaning\":\"ok def\",\"quality\":\"excellent\",\"selected_meaning\":\"to move fast\"}"}}]}`)
		case strings.Contains(content, "empty"):
			fmt.Fprint(w, `{"choices":[{"message":{"content":"{\"simple_meaning\":\"\",\"quality\":\"good\"}"}}]}`)
		default:
			fmt.Fprint(w, `{"choices":[{"message":{"content":"{\"simple_meaning\":\"to move quickly on your feet\",\"quality\":\"good\",\"selected_meaning\":\"to move fast\"}"}}]}`)
		}
	})
	m.ts = httptest.NewServer(mux)
	t.Cleanup(m.ts.Close)
	return m
}

func inRow(target, phrase string) InRow {
	return InRow{
		TargetWord: target,
		RootWord:   target,
		Phrase:     phrase,
		IPA:        "rʌn",
		Meanings:   []string{"to move fast", "to operate"},
	}
}

func rewriteRows(t *testing.T, m *mockChat, opts Options, input []InRow) ([]OutRow, error) {
	t.Helper()
	var in bytes.Buffer
	if err := jsonl.WriteAll(&in, input); err != nil {
		t.Fatal(err)
	}
	f, err := os.CreateTemp(t.TempDir(), "out")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	err = Run(strings.NewReader(in.String()), f, opts)
	if err != nil {
		return nil, err
	}
	if _, err := f.Seek(0, 0); err != nil {
		t.Fatal(err)
	}
	return jsonl.ReadAll[OutRow](f)
}

func TestHappyPath(t *testing.T) {
	m := newMockChat(t)
	rows, err := rewriteRows(t, m, Options{APIURL: m.ts.URL, APIKey: "secret", Model: "gpt-4o-mini", Concurrency: 1}, []InRow{inRow("running", "they were running.")})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	row := rows[0]
	if row.SimpleMeaning != "to move quickly on your feet" || row.Quality != "good" || row.SelectedMeaning != "to move fast" {
		t.Errorf("row = %+v", row)
	}
	if row.TargetWord != "running" || row.RootWord != "running" || row.Phrase != "they were running." || row.IPA != "rʌn" {
		t.Errorf("fields not preserved: %+v", row)
	}
	if row.Key != makeKey("running", "they were running.") {
		t.Errorf("_key = %q", row.Key)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.auth != "Bearer secret" {
		t.Errorf("Authorization header = %q", m.auth)
	}
	if m.calls != 1 {
		t.Errorf("calls = %d, want 1", m.calls)
	}
}

func TestInvalidJSON(t *testing.T) {
	m := newMockChat(t)
	rows, err := rewriteRows(t, m, Options{APIURL: m.ts.URL, Model: "gpt-4o-mini", Concurrency: 1}, []InRow{inRow("xyzzy", "a nonsense word.")})
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].SimpleMeaning != "" || rows[0].Quality != "bad" || rows[0].SelectedMeaning != "" {
		t.Errorf("invalid JSON row = %+v, want empty meaning and bad quality", rows[0])
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.calls != 3 {
		t.Errorf("calls = %d, want 3 (retries exhausted)", m.calls)
	}
}

func TestRetryThenSuccess(t *testing.T) {
	m := newMockChat(t)
	rows, err := rewriteRows(t, m, Options{APIURL: m.ts.URL, Model: "gpt-4o-mini", Concurrency: 1}, []InRow{inRow("fail500", "this row fails twice.")})
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].SimpleMeaning != "recovered after retries" || rows[0].Quality != "medium" || rows[0].SelectedMeaning != "to move fast" {
		t.Errorf("row = %+v", rows[0])
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.fail500 != 3 {
		t.Errorf("fail500 attempts = %d, want 3", m.fail500)
	}
}

func TestNonRetryableStatus(t *testing.T) {
	m := newMockChat(t)
	rows, err := rewriteRows(t, m, Options{APIURL: m.ts.URL, Model: "gpt-4o-mini", Concurrency: 1}, []InRow{inRow("forbidden", "rejected row.")})
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].SimpleMeaning != "" || rows[0].Quality != "bad" || rows[0].SelectedMeaning != "" {
		t.Errorf("row = %+v", rows[0])
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.calls != 1 {
		t.Errorf("calls = %d, want 1 (non-retryable fails immediately)", m.calls)
	}
}

func TestQualityNormalization(t *testing.T) {
	m := newMockChat(t)
	rows, err := rewriteRows(t, m, Options{APIURL: m.ts.URL, Model: "gpt-4o-mini", Concurrency: 1}, []InRow{inRow("weird", "odd quality row.")})
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].SimpleMeaning != "ok def" || rows[0].Quality != "bad" || rows[0].SelectedMeaning != "to move fast" {
		t.Errorf("row = %+v, want quality defaulted to bad", rows[0])
	}
}

func TestEmptySimpleMeaning(t *testing.T) {
	m := newMockChat(t)
	rows, err := rewriteRows(t, m, Options{APIURL: m.ts.URL, Model: "gpt-4o-mini", Concurrency: 1}, []InRow{inRow("empty", "empty meaning row.")})
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].SimpleMeaning != "" || rows[0].Quality != "bad" || rows[0].SelectedMeaning != "" {
		t.Errorf("row = %+v, want empty meaning forced to bad", rows[0])
	}
}

func TestSelectedMeaningParaphraseRejected(t *testing.T) {
	m := newMockChat(t)
	rows, err := rewriteRows(t, m, Options{APIURL: m.ts.URL, Model: "gpt-4o-mini", Concurrency: 1}, []InRow{inRow("paraphrase", "the model paraphrases the meaning.")})
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].SelectedMeaning != "" || rows[0].Quality != "bad" {
		t.Errorf("row = %+v, want unverifiable selected_meaning dropped and quality forced to bad", rows[0])
	}
	if rows[0].SimpleMeaning != "to move quickly on your feet" {
		t.Errorf("simple_meaning = %q, want kept", rows[0].SimpleMeaning)
	}
}

func TestSelectedMeaningFieldMissing(t *testing.T) {
	m := newMockChat(t)
	rows, err := rewriteRows(t, m, Options{APIURL: m.ts.URL, Model: "gpt-4o-mini", Concurrency: 1}, []InRow{inRow("no-selected", "field absent row.")})
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].SelectedMeaning != "" || rows[0].Quality != "good" {
		t.Errorf("row = %+v, want no selected_meaning and quality kept good", rows[0])
	}
}

func TestSelectedMeaningEmpty(t *testing.T) {
	m := newMockChat(t)
	rows, err := rewriteRows(t, m, Options{APIURL: m.ts.URL, Model: "gpt-4o-mini", Concurrency: 1}, []InRow{inRow("empty-selected", "empty selected row.")})
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].SelectedMeaning != "" || rows[0].Quality != "medium" {
		t.Errorf("row = %+v, want no selected_meaning and quality kept medium", rows[0])
	}
}

func TestOrdering(t *testing.T) {
	m := newMockChat(t)
	input := []InRow{
		inRow("a", "first phrase."),
		inRow("b", "second phrase."),
		inRow("c", "third phrase."),
		inRow("d", "fourth phrase."),
	}
	rows, err := rewriteRows(t, m, Options{APIURL: m.ts.URL, Model: "gpt-4o-mini", Concurrency: 4}, input)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(input) {
		t.Fatalf("got %d rows, want %d", len(rows), len(input))
	}
	for i := range input {
		if rows[i].TargetWord != input[i].TargetWord || rows[i].Phrase != input[i].Phrase {
			t.Errorf("row %d = %q/%q, want %q/%q (input order violated)", i, rows[i].TargetWord, rows[i].Phrase, input[i].TargetWord, input[i].Phrase)
		}
		if rows[i].SelectedMeaning != "to move fast" {
			t.Errorf("row %d selected_meaning = %q, want %q", i, rows[i].SelectedMeaning, "to move fast")
		}
	}
}

func TestResume(t *testing.T) {
	m := newMockChat(t)
	input := []InRow{
		inRow("a", "phrase one."),
		inRow("b", "phrase two."),
		inRow("c", "phrase three."),
		inRow("d", "phrase four."),
	}
	dir := t.TempDir()
	path := dir + "/out.jsonl"
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	existing := []OutRow{
		{TargetWord: "a", RootWord: "a", Phrase: "phrase one.", IPA: "x", Meanings: nil, SimpleMeaning: "old", Quality: "good", Key: makeKey("a", "phrase one.")},
		{TargetWord: "c", RootWord: "c", Phrase: "phrase three.", IPA: "x", Meanings: nil, SimpleMeaning: "old", Quality: "good", Key: makeKey("c", "phrase three.")},
	}
	if err := jsonl.WriteAll(f, existing); err != nil {
		t.Fatal(err)
	}
	var in bytes.Buffer
	if err := jsonl.WriteAll(&in, input); err != nil {
		t.Fatal(err)
	}
	if err := Run(strings.NewReader(in.String()), f, Options{APIURL: m.ts.URL, Model: "gpt-4o-mini", Concurrency: 2, Resume: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Seek(0, 0); err != nil {
		t.Fatal(err)
	}
	rows, err := jsonl.ReadAll[OutRow](f)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 4 {
		t.Fatalf("resume produced %d rows, want 4", len(rows))
	}
	if rows[0].SimpleMeaning != "old" || rows[1].SimpleMeaning != "old" {
		t.Errorf("pre-existing rows changed: %+v", rows[:2])
	}
	if rows[0].SelectedMeaning != "" || rows[1].SelectedMeaning != "" {
		t.Errorf("pre-existing rows lost backward compat (selected_meaning): %+v", rows[:2])
	}
	if rows[2].TargetWord != "b" || rows[3].TargetWord != "d" {
		t.Errorf("appended rows = %q, %q, want b then d", rows[2].TargetWord, rows[3].TargetWord)
	}
	if rows[2].SimpleMeaning != "to move quickly on your feet" || rows[3].SimpleMeaning != "to move quickly on your feet" {
		t.Errorf("new rows not rewritten: %+v", rows[2:])
	}
	if rows[2].SelectedMeaning != "to move fast" || rows[3].SelectedMeaning != "to move fast" {
		t.Errorf("new rows selected_meaning = %q, %q, want both %q", rows[2].SelectedMeaning, rows[3].SelectedMeaning, "to move fast")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.calls != 2 {
		t.Errorf("calls = %d, want 2 (only unprocessed rows)", m.calls)
	}
}

func TestKeyDeterminism(t *testing.T) {
	sum := sha256.Sum256([]byte("running\x00they were running."))
	want := hex.EncodeToString(sum[:])
	if got := makeKey("running", "they were running."); got != want {
		t.Errorf("makeKey = %q, want %q", got, want)
	}
}

func TestMatchMeaning(t *testing.T) {
	tests := []struct {
		name     string
		raw      string
		meanings []string
		want     string
		wantOK   bool
	}{
		{"exact match", "to move fast", []string{"to move fast", "to operate"}, "to move fast", true},
		{"parentheses stripped", "to make small adjustments to something until it is optimal", []string{"to make small adjustments to (something) until it is optimal"}, "to make small adjustments to (something) until it is optimal", true},
		{"capitalization and punctuation", "To Move, Fast!", []string{"to move fast"}, "to move fast", true},
		{"word reorder", "fast to move", []string{"to move fast"}, "to move fast", true},
		{"paraphrase different words", "to go quickly", []string{"to move fast"}, "", false},
		{"empty raw", "", []string{"to move fast"}, "", false},
		{"whitespace only raw", "   ", []string{"to move fast"}, "", false},
		{"subset of meaning", "to operate", []string{"to operate or function"}, "to operate or function", true},
		{"tie keeps first entry", "a b c", []string{"a b c", "b c a"}, "a b c", true},
		{"empty meanings", "to move fast", nil, "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := matchMeaning(tt.raw, tt.meanings)
			if got != tt.want || ok != tt.wantOK {
				t.Errorf("matchMeaning(%q, %v) = (%q, %v), want (%q, %v)", tt.raw, tt.meanings, got, ok, tt.want, tt.wantOK)
			}
		})
	}
}

func TestNormalize(t *testing.T) {
	tests := []struct{ in, want string }{
		{"Hello World", "hello world"},
		{"To move, FAST!", "to move fast"},
		{"to make small adjustments to (something) until it is optimal", "to make small adjustments to something until it is optimal"},
		{"to   move   fast", "to move fast"},
		{"what's up?", "whats up"},
		{"", ""},
		{"!!!", ""},
	}
	for _, tt := range tests {
		if got := normalize(tt.in); got != tt.want {
			t.Errorf("normalize(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestMissingOptions(t *testing.T) {
	f, _ := os.CreateTemp(t.TempDir(), "out")
	defer f.Close()
	if err := Run(strings.NewReader(""), f, Options{Model: "m"}); err == nil {
		t.Fatal("expected error when api-url is missing")
	}
	if err := Run(strings.NewReader(""), f, Options{APIURL: "http://x"}); err == nil {
		t.Fatal("expected error when model is missing")
	}
}
