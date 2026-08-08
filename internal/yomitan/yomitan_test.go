package yomitan

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	espeak = func(context.Context, string) string { return "" }
	os.Exit(m.Run())
}

func newMockServer(t *testing.T) *httptest.Server {
	t.Helper()
	fixture, err := os.ReadFile("testdata/bank_kty.json")
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/termEntries", func(w http.ResponseWriter, r *http.Request) {
		var body struct{ Term string }
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		switch body.Term {
		case "run":
			fmt.Fprint(w, `{"dictionaryEntries":[`+
				`{"headwords":[{"term":"run","reading":"rʌn"}],"definitions":[`+
				`{"dictionary":"CambridgeV1","entries":["verb\n\tto move along faster than walking by taking quick steps: \n\t· The children had to run to keep up.\n\n\tverb\n\tto operate or function: \n\t· The engine runs smoothly.\n\n\trun on the spot UK\n\tverb\n\tto move your legs as if running while staying in one place: \n\t· I run on the spot to warm up.\n"]},`+
				`{"dictionary":"New Oxford American Dict Yomi-v3","entries":["run  | rənrən |\nverb (runs, running)\n1 move at a speed faster than a walk:  the dog ran across the road.\n2 be in charge of; manage:  Andrea runs her own business.\n"]}`+
				`]}]}`)
		case "bank":
			fmt.Fprintf(w, `{"dictionaryEntries":[`+
				`{"headwords":[{"term":"bank","reading":"bæŋk"}],"definitions":[]},%s]}`, fixture)
		case "xyzzy":
			fmt.Fprint(w, `{"dictionaryEntries":[]}`)
		default:
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprint(w, `{"error":"boom"}`)
		}
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts
}

func TestCambridgeRun(t *testing.T) {
	ts := newMockServer(t)
	c := NewClient(ts.URL, time.Second)
	entry, err := c.TermEntries(context.Background(), "run")
	if err != nil {
		t.Fatal(err)
	}
	if entry.IPA != "rʌn" {
		t.Errorf("IPA = %q, want rʌn", entry.IPA)
	}
	want := []string{
		"to move along faster than walking by taking quick steps",
		"to operate or function",
		"to move your legs as if running while staying in one place",
		"move at a speed faster than a walk",
		"be in charge of; manage",
	}
	for _, m := range want {
		if !contains(entry.Meanings, m) {
			t.Errorf("meanings missing %q; got %v", m, entry.Meanings)
		}
	}
	for _, m := range entry.Meanings {
		if m == "the children had to run to keep up" {
			t.Errorf("pure example sentence leaked into meanings: %v", entry.Meanings)
		}
	}
}

func TestStructuredBank(t *testing.T) {
	ts := newMockServer(t)
	c := NewClient(ts.URL, time.Second)
	entry, err := c.TermEntries(context.Background(), "bank")
	if err != nil {
		t.Fatal(err)
	}
	if entry.IPA != "bæŋk" {
		t.Errorf("IPA = %q, want bæŋk", entry.IPA)
	}
	if len(entry.Meanings) < 5 {
		t.Fatalf("got %d meanings, want at least 5: %v", len(entry.Meanings), entry.Meanings)
	}
	if len(entry.Meanings) > 12 {
		t.Errorf("got %d meanings, want cap of 12", len(entry.Meanings))
	}
	want := []string{
		"an institution where one can place and borrow money and take care of financial affairs",
		"(hydrology) an edge of river, lake, or other watercourse",
	}
	for _, m := range want {
		if !contains(entry.Meanings, m) {
			t.Errorf("meanings missing %q", m)
		}
	}
}

func TestEmptyEntry(t *testing.T) {
	ts := newMockServer(t)
	c := NewClient(ts.URL, time.Second)
	entry, err := c.TermEntries(context.Background(), "xyzzy")
	if err != nil {
		t.Fatal(err)
	}
	if entry.IPA != "" {
		t.Errorf("IPA = %q, want empty", entry.IPA)
	}
	if len(entry.Meanings) != 0 {
		t.Errorf("meanings = %v, want empty", entry.Meanings)
	}
}

func TestServerError(t *testing.T) {
	ts := newMockServer(t)
	c := NewClient(ts.URL, time.Second)
	if _, err := c.TermEntries(context.Background(), "bogus"); err == nil {
		t.Fatal("expected error for 500 response")
	}
}

func TestUnreachable(t *testing.T) {
	c := NewClient("http://127.0.0.1:1", time.Second)
	_, err := c.TermEntries(context.Background(), "run")
	if err == nil {
		t.Fatal("expected error for unreachable server")
	}
	if !IsUnreachable(err) {
		t.Errorf("IsUnreachable(%v) = false", err)
	}
	if !strings.Contains(err.Error(), "Yomitan server unreachable at http://127.0.0.1:1") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestEspeakFallback(t *testing.T) {
	old := espeak
	espeak = func(context.Context, string) string { return "stub-ipa" }
	t.Cleanup(func() { espeak = old })
	ts := newMockServer(t)
	c := NewClient(ts.URL, time.Second)
	entry, err := c.TermEntries(context.Background(), "xyzzy")
	if err != nil {
		t.Fatal(err)
	}
	if entry.IPA != "stub-ipa" {
		t.Errorf("IPA = %q, want stub-ipa from espeak fallback", entry.IPA)
	}
}

func TestDedupAndNormalization(t *testing.T) {
	glosses := dedupCap([]string{
		"A fund from deposits.",
		"A fund from deposits!",
		"  • lead bullet",
		"- lead bullet",
		"Empty.",
	}, 12)
	if len(glosses) != 3 {
		t.Fatalf("got %d glosses, want 3 (a fund from deposits, lead bullet, empty)", len(glosses))
	}
	if glosses[0] != "a fund from deposits" {
		t.Errorf("gloss[0] = %q", glosses[0])
	}
	if glosses[1] != "lead bullet" {
		t.Errorf("gloss[1] = %q", glosses[1])
	}
	if glosses[2] != "empty" {
		t.Errorf("gloss[2] = %q", glosses[2])
	}
}

func TestLooksLikeIPA(t *testing.T) {
	for _, s := range []string{"rʌn", "/rʌn/", "[rʌn]", "ˈrʌn.ɪŋ"} {
		if !looksLikeIPA(s) {
			t.Errorf("looksLikeIPA(%q) = false, want true", s)
		}
	}
	for _, s := range []string{"run", "bank", "Bank", "running", ""} {
		if looksLikeIPA(s) {
			t.Errorf("looksLikeIPA(%q) = true, want false", s)
		}
	}
}

func TestCambridgeMultiPOSChunk(t *testing.T) {
	s := "adjective\n\ta language when it was in an early stage in its development\n\t\n\tadjective\n\t(especially of a friend) known for a long time: \n\t· example sentence."
	glosses := dedupCap(extractPlain(s), 12)
	want := []string{
		"a language when it was in an early stage in its development",
		"(especially of a friend) known for a long time",
	}
	if len(glosses) != len(want) {
		t.Fatalf("got %v, want %v", glosses, want)
	}
	for i := range want {
		if glosses[i] != want[i] {
			t.Errorf("gloss[%d] = %q, want %q", i, glosses[i], want[i])
		}
	}
}

func TestNewOxfordColonNewlineExample(t *testing.T) {
	s := "old  | ōld |\nadjective (older)\n1 having lived for a long time:  example here.\n2 [attributive] belonging only or chiefly to the past; former or previous:\nvaluation under the old rating system was inexact."
	glosses := dedupCap(extractPlain(s), 12)
	want := []string{
		"having lived for a long time",
		"[attributive] belonging only or chiefly to the past; former or previous",
	}
	if len(glosses) != len(want) {
		t.Fatalf("got %v, want %v", glosses, want)
	}
	for i := range want {
		if glosses[i] != want[i] {
			t.Errorf("gloss[%d] = %q, want %q", i, glosses[i], want[i])
		}
	}
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}
