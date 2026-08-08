package augment

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/kafo-play-box/shisui/internal/jsonl"
	"github.com/kafo-play-box/shisui/internal/yomitan"
)

func TestMain(m *testing.M) {
	retryDelay = time.Millisecond
	os.Exit(m.Run())
}

func newMockYomitan(t *testing.T) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/termEntries", func(w http.ResponseWriter, r *http.Request) {
		var body struct{ Term string }
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		switch body.Term {
		case "run":
			fmt.Fprint(w, `{"dictionaryEntries":[{"headwords":[{"term":"run","reading":"rʌn"}],"definitions":[{"dictionary":"CambridgeV1","entries":["verb\n\tto move along faster than walking: \n\t· The children had to run.\n"]}]}]}`)
		case "bank":
			fmt.Fprint(w, `{"dictionaryEntries":[{"headwords":[{"term":"bank","reading":"bæŋk"}],"definitions":[{"dictionary":"kty-en-en","entries":[{"type":"structured-content","content":[{"tag":"ol","data":{"content":"glosses"},"content":[{"tag":"li","content":[{"tag":"div","content":["An institution where one can place and borrow money."]}]}]}]}]}]}]}`)
		case "xyzzy":
			fmt.Fprint(w, `{"dictionaryEntries":[{"headwords":[{"term":"xyzzy","reading":"ˈzɪzi"}],"definitions":[]}]}`)
		default:
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprint(w, `{"error":"boom"}`)
		}
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts
}

func input(rows []InRow) *strings.Reader {
	var buf bytes.Buffer
	if err := jsonl.WriteAll(&buf, rows); err != nil {
		panic(err)
	}
	return strings.NewReader(buf.String())
}

func captureStderr(t *testing.T, f func() error) (string, error) {
	t.Helper()
	old := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stderr = w
	runErr := f()
	w.Close()
	os.Stderr = old
	data, err := io.ReadAll(r)
	r.Close()
	if err != nil {
		t.Fatal(err)
	}
	return string(data), runErr
}

func TestRun(t *testing.T) {
	ts := newMockYomitan(t)
	client := yomitan.NewClient(ts.URL, time.Second)
	rows := []InRow{
		{TargetWord: "running", RootWord: "run", Phrase: "they were running."},
		{TargetWord: "bank", RootWord: "bank", Phrase: "the bank of the river."},
		{TargetWord: "xyzzy", RootWord: "xyzzy", Phrase: "a nonsense word."},
		{TargetWord: "bogus", RootWord: "bogus", Phrase: "a failing word."},
	}
	var out bytes.Buffer
	if err := Run(input(rows), &out, Options{YomitanClient: client, Concurrency: 4}); err != nil {
		t.Fatal(err)
	}
	got, err := jsonl.ReadAll[OutRow](&out)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d rows, want 2 (xyzzy/bogus dropped for empty meanings)", len(got))
	}
	if got[0].TargetWord != "running" || got[0].Phrase != "they were running." {
		t.Errorf("row 0 out of order: %+v", got[0])
	}
	if got[1].TargetWord != "bank" || got[1].Phrase != "the bank of the river." {
		t.Errorf("row 1 out of order: %+v", got[1])
	}
	if got[0].IPA != "rʌn" || !contains(got[0].Meanings, "to move along faster than walking") {
		t.Errorf("run row = %+v", got[0])
	}
	if got[1].IPA != "bæŋk" || !contains(got[1].Meanings, "an institution where one can place and borrow money") {
		t.Errorf("bank row = %+v", got[1])
	}
}

func TestOrdering(t *testing.T) {
	ts := newMockYomitan(t)
	client := yomitan.NewClient(ts.URL, time.Second)
	rows := []InRow{
		{TargetWord: "a", RootWord: "run", Phrase: "first."},
		{TargetWord: "b", RootWord: "xyzzy", Phrase: "second."},
		{TargetWord: "c", RootWord: "bank", Phrase: "third."},
		{TargetWord: "d", RootWord: "run", Phrase: "fourth."},
	}
	var out bytes.Buffer
	if err := Run(input(rows), &out, Options{YomitanClient: client, Concurrency: 4}); err != nil {
		t.Fatal(err)
	}
	got, err := jsonl.ReadAll[OutRow](&out)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("got %d rows, want 3 (xyzzy dropped)", len(got))
	}
	want := []string{"a", "c", "d"}
	for i, w := range want {
		if got[i].TargetWord != w {
			t.Errorf("row %d = %q, want %q (survivors keep input order)", i, got[i].TargetWord, w)
		}
	}
}

func TestEmptyMeaningsDropped(t *testing.T) {
	ts := newMockYomitan(t)
	client := yomitan.NewClient(ts.URL, time.Second)
	rows := []InRow{
		{TargetWord: "running", RootWord: "run", Phrase: "they were running."},
		{TargetWord: "xyzzy", RootWord: "xyzzy", Phrase: "a nonsense word."},
	}
	var out bytes.Buffer
	data, err := captureStderr(t, func() error {
		return Run(input(rows), &out, Options{YomitanClient: client, Concurrency: 4})
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := jsonl.ReadAll[OutRow](&out)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d rows, want 1 (xyzzy dropped)", len(got))
	}
	if got[0].TargetWord != "running" {
		t.Errorf("row = %+v, want running", got[0])
	}
	if !strings.Contains(data, "skipped 1 row(s) with empty meanings") {
		t.Errorf("stderr missing skip log: %q", data)
	}
}

func TestEmptyMeaningsAllDropped(t *testing.T) {
	ts := newMockYomitan(t)
	client := yomitan.NewClient(ts.URL, time.Second)
	rows := []InRow{{TargetWord: "xyzzy", RootWord: "xyzzy", Phrase: "a nonsense word."}}
	var out bytes.Buffer
	data, err := captureStderr(t, func() error {
		return Run(input(rows), &out, Options{YomitanClient: client, Concurrency: 2})
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.Len() != 0 {
		t.Errorf("expected empty output, got %q", out.String())
	}
	if !strings.Contains(data, "skipped 1 row(s) with empty meanings") {
		t.Errorf("stderr missing skip log: %q", data)
	}
}

func TestUnreachableIsFatal(t *testing.T) {
	client := yomitan.NewClient("http://127.0.0.1:1", time.Second)
	rows := []InRow{{TargetWord: "run", RootWord: "run", Phrase: "a phrase."}}
	var out bytes.Buffer
	err := Run(input(rows), &out, Options{YomitanClient: client, Concurrency: 2})
	if err == nil {
		t.Fatal("expected error for unreachable server")
	}
	if !strings.Contains(err.Error(), "Yomitan server unreachable") {
		t.Errorf("unexpected error: %v", err)
	}
	if out.Len() != 0 {
		t.Errorf("output written despite fatal error: %q", out.String())
	}
}

func TestEmptyInput(t *testing.T) {
	ts := newMockYomitan(t)
	client := yomitan.NewClient(ts.URL, time.Second)
	var out bytes.Buffer
	if err := Run(strings.NewReader(""), &out, Options{YomitanClient: client, Concurrency: 4}); err != nil {
		t.Fatal(err)
	}
	if out.Len() != 0 {
		t.Errorf("expected empty output, got %q", out.String())
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
