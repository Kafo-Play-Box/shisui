package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOutput_StdoutNoGuard(t *testing.T) {
	w, closeOut, err := output("", false)
	if err != nil {
		t.Fatal(err)
	}
	if w != os.Stdout {
		t.Errorf("output(\"\") = %v, want os.Stdout", w)
	}
	closeOut()
}

func TestOutput_NewFileNoForce(t *testing.T) {
	p := filepath.Join(t.TempDir(), "out.jsonl")
	if _, closeOut, err := output(p, false); err != nil {
		t.Fatal(err)
	} else {
		closeOut()
	}
	if _, err := os.Stat(p); err != nil {
		t.Errorf("output file not created: %v", err)
	}
}

func TestOutput_ExistingNoForce_Error(t *testing.T) {
	p := filepath.Join(t.TempDir(), "out.jsonl")
	if err := os.WriteFile(p, []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	_, _, err := output(p, false)
	if err == nil {
		t.Fatal("expected error for existing file without --force")
	}
	if !strings.Contains(err.Error(), "use --force") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestOutput_ExistingWithForce_OK(t *testing.T) {
	p := filepath.Join(t.TempDir(), "out.jsonl")
	if err := os.WriteFile(p, []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, closeOut, err := output(p, true); err != nil {
		t.Fatalf("output with --force failed: %v", err)
	} else {
		closeOut()
	}
}

func TestScrapeDirectoryMode(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out.txt")
	dir := filepath.Join("internal", "scrape", "testdata", "directory_fixture")
	if err := cmdScrape([]string{"-o", out, dir}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	// Sorted order: page1.html before sub/page2.html, one blank line between.
	if want := "Page one.\n\nPage two."; string(data) != want {
		t.Errorf("got %q, want %q", data, want)
	}
}

func TestScrapeURLMode(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "<p>Hello from the web.</p>")
	}))
	defer ts.Close()
	out := filepath.Join(t.TempDir(), "out.txt")
	if err := cmdScrape([]string{"-o", out, ts.URL}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if want := "Hello from the web."; string(data) != want {
		t.Errorf("got %q, want %q", data, want)
	}
}

func TestScrapeURLMode_Error(t *testing.T) {
	ts := httptest.NewServer(http.NotFoundHandler())
	defer ts.Close()
	err := cmdScrape([]string{ts.URL + "/missing"})
	if err == nil {
		t.Fatal("expected error for 404")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRewrite_ForceAndResume_Error(t *testing.T) {
	p := filepath.Join(t.TempDir(), "out.jsonl")
	err := cmdRewrite([]string{"--api-url", "http://127.0.0.1:1", "--output", p, "--force", "--resume"})
	if err == nil {
		t.Fatal("expected error for --force with --resume")
	}
	if !strings.Contains(err.Error(), "cannot be used together") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRewrite_OutputExistsNoFlags_Error(t *testing.T) {
	p := filepath.Join(t.TempDir(), "out.jsonl")
	if err := os.WriteFile(p, []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	err := cmdRewrite([]string{"--api-url", "http://127.0.0.1:1", "--output", p})
	if err == nil {
		t.Fatal("expected error for existing output without --force/--resume")
	}
	if !strings.Contains(err.Error(), "use --force") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRewrite_Force_Truncates(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"choices":[{"message":{"content":"{\"simple_meaning\":\"a simple meaning\",\"quality\":\"good\"}"}}]}`)
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	in := filepath.Join(t.TempDir(), "in.jsonl")
	if err := os.WriteFile(in, []byte(`{"target_word":"cat","root_word":"cat","phrase":"the cat ran.","ipa":"","meanings":["a cat"]}`+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "out.jsonl")
	if err := os.WriteFile(out, []byte("OLD JUNK\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := cmdRewrite([]string{"--api-url", ts.URL, "--force", "--output", out, in}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "OLD JUNK") {
		t.Errorf("output not truncated: %q", data)
	}
	if !strings.Contains(string(data), "a simple meaning") {
		t.Errorf("output missing rewritten row: %q", data)
	}
}
