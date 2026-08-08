package main

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
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

// writeFixture writes rel under dir, creating parent directories.
func writeFixture(t *testing.T, dir, rel, content string) {
	t.Helper()
	p := filepath.Join(dir, rel)
	if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
}

// zimFixtureDir populates dir like a zimdump dump: extensionless articles,
// nested subdirs for slash paths, an _exceptions/ dir, resource dirs, and a
// dump_errors.log. Returns the dir.
func zimFixtureDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFixture(t, dir, "ArticleOne", "<p>First article.</p>")
	writeFixture(t, dir, "ArticleTwo", "<p>Second article.</p>")
	writeFixture(t, dir, filepath.Join("Medicine", "Bone"), "<p>Bone article.</p>")
	writeFixture(t, dir, filepath.Join("_exceptions", "Encoded%2fPath"), "<p>Encoded path article.</p>")
	writeFixture(t, dir, "dump_errors.log", "some error log text")
	// Resource dirs come from non-A namespaces; their content must not leak.
	writeFixture(t, dir, filepath.Join("_res_", "junk.css"), "<p>resource junk css</p>")
	writeFixture(t, dir, filepath.Join("_mw_", "junk.js"), "<p>resource junk js</p>")
	return dir
}

func TestScrapeAllFiles(t *testing.T) {
	var buf bytes.Buffer
	if err := scrapeAllFiles(zimFixtureDir(t), &buf); err != nil {
		t.Fatal(err)
	}
	data := buf.String()
	for _, want := range []string{"First article.", "Second article.", "Bone article.", "Encoded path article."} {
		if !strings.Contains(data, want) {
			t.Errorf("output missing %q:\n%s", want, data)
		}
	}
	for _, bad := range []string{"some error log text", "resource junk css", "resource junk js"} {
		if strings.Contains(data, bad) {
			t.Errorf("output contains %q:\n%s", bad, data)
		}
	}
}

func TestScrapeZimMode(t *testing.T) {
	orig := zimdumpRun
	zimdumpRun = func(zimPath string, w io.Writer) error {
		return scrapeAllFiles(zimFixtureDir(t), w)
	}
	defer func() { zimdumpRun = orig }()
	out := filepath.Join(t.TempDir(), "out.txt")
	if err := cmdScrape([]string{"-o", out, "test.zim"}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"First article.", "Second article.", "Bone article.", "Encoded path article."} {
		if !strings.Contains(string(data), want) {
			t.Errorf("output missing %q: %q", want, data)
		}
	}
}

func TestScrapeZimMode_RedirectStubSkipped(t *testing.T) {
	orig := zimdumpRun
	zimdumpRun = func(zimPath string, w io.Writer) error {
		dir := t.TempDir()
		writeFixture(t, dir, "ArticleOne", "<p>Zim article one.</p>")
		writeFixture(t, dir, "RedirectStub", `<meta http-equiv="refresh" content="0;url=/A/Target">`)
		return scrapeAllFiles(dir, w)
	}
	defer func() { zimdumpRun = orig }()
	out := filepath.Join(t.TempDir(), "out.txt")
	if err := cmdScrape([]string{"-o", out, "test.zim"}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "Zim article one.") {
		t.Errorf("output missing article: %q", data)
	}
	for _, bad := range []string{"Target", "refresh", "RedirectStub"} {
		if strings.Contains(string(data), bad) {
			t.Errorf("output contains stub content %q: %q", bad, data)
		}
	}
}

func TestScrapeZimMode_NoZimdump_Error(t *testing.T) {
	t.Setenv("PATH", "/nonexistent")
	var buf bytes.Buffer
	err := zimdumpRun("test.zim", &buf)
	if err == nil {
		t.Fatal("expected error when zimdump is not on PATH")
	}
	if !strings.Contains(err.Error(), "zimdump not found") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestScrapeZimIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping ZIM integration test in -short mode")
	}
	if _, err := exec.LookPath("zimdump"); err != nil {
		t.Skip("zimdump not installed")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("cannot resolve home dir: %v", err)
	}
	zimPath := filepath.Join(home, ".local/share/kiwix-desktop/wikipedia_en_100_nopic_2026-07.zim")
	if _, err := os.Stat(zimPath); err != nil {
		t.Skipf("ZIM file not present: %v", err)
	}
	out := filepath.Join(t.TempDir(), "out.txt")
	if err := cmdScrape([]string{"-o", out, zimPath}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 {
		t.Fatal("ZIM scrape produced no output")
	}
	if got := strings.Count(string(data), "\n\n"); got < 10 {
		t.Errorf("blank-line blocks = %d, want >= 10", got)
	}
	junk := regexp.MustCompile(`vector-toc|mw-editsection|catlinks|Jump to content|Related articles`)
	if m := junk.Find(data); m != nil {
		t.Errorf("output contains chrome junk %q", m)
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
