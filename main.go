package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/kafo-play-box/shisui/internal/augment"
	"github.com/kafo-play-box/shisui/internal/extract"
	"github.com/kafo-play-box/shisui/internal/lemma"
	"github.com/kafo-play-box/shisui/internal/rewrite"
	"github.com/kafo-play-box/shisui/internal/scrape"
	"github.com/kafo-play-box/shisui/internal/yomitan"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "scrape":
		err = cmdScrape(os.Args[2:])
	case "extract":
		err = cmdExtract(os.Args[2:])
	case "augment":
		err = cmdAugment(os.Args[2:])
	case "rewrite":
		err = cmdRewrite(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "shisui:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `shisui <command> [flags]

commands:
  scrape   [--force] [-o PATH] <input.html | input-dir | input.zim | URL
  extract  [--lemma PATH] [--min-words N] [--max-words N] [--min-target-len N] [--force] [-o PATH] <input.txt
  augment  [--yomitan-url URL] [--yomitan-timeout DUR] [--concurrency N] [--force] [-o PATH] <input.jsonl
  rewrite  --api-url URL [--api-key KEY] [--model NAME] [--concurrency N] [--resume] [--force] -o PATH <input.jsonl

input is read from stdin, or from a positional file argument.
scrape additionally accepts a directory, a .zim file, or an http(s) URL.
`)
}

func cmdScrape(args []string) error {
	fs := flag.NewFlagSet("scrape", flag.ContinueOnError)
	outPath := fs.String("output", "", "output file (default: stdout)")
	fs.StringVar(outPath, "o", "", "output file (default: stdout)")
	force := fs.Bool("force", false, "overwrite an existing output file")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() > 1 {
		return errors.New("too many input files")
	}
	out, closeOut, err := output(*outPath, *force)
	if err != nil {
		return err
	}
	defer closeOut()
	if fs.NArg() == 0 {
		return scrape.Run(os.Stdin, out)
	}
	arg := fs.Arg(0)
	switch {
	case strings.HasPrefix(arg, "http://"), strings.HasPrefix(arg, "https://"):
		return scrapeURL(arg, out)
	case strings.HasSuffix(strings.ToLower(arg), ".zim"):
		return zimdumpRun(arg, out)
	default:
		fi, err := os.Stat(arg)
		if err != nil {
			return err
		}
		if fi.IsDir() {
			return scrapeDir(arg, out)
		}
		f, err := os.Open(arg)
		if err != nil {
			return err
		}
		defer f.Close()
		return scrape.Run(f, out)
	}
}

// scrapeURL fetches url and scrapes the response body. Non-2xx responses are
// errors.
func scrapeURL(url string, w io.Writer) error {
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return fmt.Errorf("scrape: %s returned status %s", url, resp.Status)
	}
	return scrape.Run(resp.Body, w)
}

// scrapeDir walks dir for *.html/*.htm files in sorted order and scrapes each
// into w, one blank line between files. A file that fails to open or parse is
// logged to stderr and skipped; the walk continues. Only one file's output is
// held in memory at a time.
func scrapeDir(dir string, w io.Writer) error {
	var files []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		ext := filepath.Ext(path)
		if ext == ".html" || ext == ".htm" {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return err
	}
	return scrapeFileList(files, w)
}

// scrapeFileList sorts files and scrapes each into w, one blank line between
// documents. A file that fails to open or parse is logged to stderr and
// skipped; a file that scrapes to nothing (e.g. a redirect stub) is skipped.
// Only one file's output is held in memory at a time. Returns the first write
// error.
func scrapeFileList(files []string, w io.Writer) error {
	sort.Strings(files)
	first := true
	for _, path := range files {
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "scrape: %s: %v\n", path, err)
			continue
		}
		var buf bytes.Buffer
		if err := scrape.Run(f, &buf); err != nil {
			f.Close()
			fmt.Fprintf(os.Stderr, "scrape: %s: %v\n", path, err)
			continue
		}
		f.Close()
		if buf.Len() == 0 {
			continue
		}
		if !first {
			if err := writeAll(w, []byte("\n\n")); err != nil {
				return err
			}
		}
		first = false
		if err := writeAll(w, buf.Bytes()); err != nil {
			return err
		}
	}
	return nil
}

// zimNamespace is the ZIM article namespace scraped (Kiwix/Wikipedia). It is
// passed to zimdump as --ns=A; zim-tools 3.7.0 ignores the filter on dump and
// writes every namespace, so scrapeAllFiles skips the resource dirs.
const zimNamespace = "A"

// zimdumpRun dumps zimPath with the zimdump binary and scrapes every article
// in the dump into w. Overridable in tests.
var zimdumpRun = func(zimPath string, w io.Writer) error {
	absPath, err := filepath.Abs(zimPath)
	if err != nil {
		return err
	}
	if _, err := exec.LookPath("zimdump"); err != nil {
		return errors.New("zimdump not found; install zim-tools (e.g. pacman -S zim-tools)")
	}
	tmpDir, err := os.MkdirTemp("", "shisui-zim-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)
	cmd := exec.Command("zimdump", "dump", "--dir="+tmpDir, "--ns="+zimNamespace, absPath)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if len(msg) > 500 {
			msg = msg[len(msg)-500:]
		}
		return fmt.Errorf("zimdump: %v: %s", err, msg)
	}
	return scrapeAllFiles(tmpDir, w)
}

// scrapeAllFiles walks dir for every regular file except dump_errors.log and
// scrapes each into w via scrapeFileList. Used for zimdump dump output
// directories, whose articles are extensionless and nested (slash paths in
// subdirs, escaped paths in _exceptions/). The Kiwix resource directories
// (_res_, _mw_, _assets_, _webp_) hold CSS/JS/images from other namespaces;
// zimdump's --ns=A filter is a no-op in zim-tools 3.7.0, so they appear in
// the dump and are skipped here — their content is not article prose.
func scrapeAllFiles(dir string, w io.Writer) error {
	var files []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == "_res_" || d.Name() == "_mw_" || d.Name() == "_assets_" || d.Name() == "_webp_" {
				return filepath.SkipDir
			}
			return nil
		}
		if d.Name() == "dump_errors.log" {
			return nil
		}
		files = append(files, path)
		return nil
	})
	if err != nil {
		return err
	}
	return scrapeFileList(files, w)
}

// writeAll writes b to w, looping over partial writes until all bytes are
// written or an error occurs.
func writeAll(w io.Writer, b []byte) error {
	for len(b) > 0 {
		n, err := w.Write(b)
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
		b = b[n:]
	}
	return nil
}

func cmdExtract(args []string) error {
	fs := flag.NewFlagSet("extract", flag.ContinueOnError)
	lemmaPath := fs.String("lemma", "", "custom lemma file (default: embedded)")
	fs.StringVar(lemmaPath, "l", "", "custom lemma file (default: embedded)")
	minWords := fs.Int("min-words", 10, "drop phrases with fewer words than this")
	maxWords := fs.Int("max-words", 80, "truncate phrases longer than this many words")
	minTargetLen := fs.Int("min-target-len", 3, "drop candidate targets shorter than this many characters")
	outPath := fs.String("output", "", "output file (default: stdout)")
	fs.StringVar(outPath, "o", "", "output file (default: stdout)")
	force := fs.Bool("force", false, "overwrite an existing output file")
	if err := fs.Parse(args); err != nil {
		return err
	}
	in, err := input(fs)
	if err != nil {
		return err
	}
	defer in.Close()
	out, closeOut, err := output(*outPath, *force)
	if err != nil {
		return err
	}
	defer closeOut()
	var data []byte
	if *lemmaPath != "" {
		data, err = os.ReadFile(*lemmaPath)
	} else {
		data = EmbeddedLemma
	}
	if err != nil {
		return err
	}
	db, err := lemma.Parse(bytes.NewReader(data))
	if err != nil {
		return err
	}
	return extract.Run(in, out, extract.Options{LemmaDB: db, MinWords: *minWords, MaxWords: *maxWords, MinTargetLen: *minTargetLen})
}

func cmdAugment(args []string) error {
	fs := flag.NewFlagSet("augment", flag.ContinueOnError)
	baseURL := fs.String("yomitan-url", "http://127.0.0.1:19633", "yomitan server base URL")
	timeout := fs.Duration("yomitan-timeout", 5*time.Second, "per-request yomitan timeout")
	concurrency := fs.Int("concurrency", 4, "parallel yomitan requests")
	outPath := fs.String("output", "", "output file (default: stdout)")
	fs.StringVar(outPath, "o", "", "output file (default: stdout)")
	force := fs.Bool("force", false, "overwrite an existing output file")
	if err := fs.Parse(args); err != nil {
		return err
	}
	in, err := input(fs)
	if err != nil {
		return err
	}
	defer in.Close()
	out, closeOut, err := output(*outPath, *force)
	if err != nil {
		return err
	}
	defer closeOut()
	client := yomitan.NewClient(*baseURL, *timeout)
	return augment.Run(in, out, augment.Options{YomitanClient: client, Concurrency: *concurrency})
}

func cmdRewrite(args []string) error {
	fs := flag.NewFlagSet("rewrite", flag.ContinueOnError)
	apiURL := fs.String("api-url", "", "OpenAI-compatible chat completions base URL")
	apiKey := fs.String("api-key", "", "API key (default: SHISUI_API_KEY env)")
	model := fs.String("model", "gpt-4o-mini", "model name")
	concurrency := fs.Int("concurrency", 4, "parallel API workers")
	resume := fs.Bool("resume", false, "skip rows already present in the output file")
	force := fs.Bool("force", false, "truncate and rewrite the output file from scratch")
	outPath := fs.String("output", "", "output file (required)")
	fs.StringVar(outPath, "o", "", "output file (required)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *apiURL == "" {
		return errors.New("rewrite: --api-url is required")
	}
	if *outPath == "" {
		return errors.New("rewrite: --output/-o is required")
	}
	if *force && *resume {
		return errors.New("rewrite: --force and --resume cannot be used together")
	}
	in, err := input(fs)
	if err != nil {
		return err
	}
	defer in.Close()
	key := *apiKey
	if key == "" {
		key = os.Getenv("SHISUI_API_KEY")
	}
	var f *os.File
	switch {
	case *resume:
		f, err = os.OpenFile(*outPath, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0644)
	case *force:
		f, err = os.OpenFile(*outPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	default:
		if _, statErr := os.Stat(*outPath); statErr == nil {
			return fmt.Errorf("output %s exists, use --force to overwrite", *outPath)
		}
		f, err = os.Create(*outPath)
	}
	if err != nil {
		return err
	}
	defer f.Close()
	return rewrite.Run(in, f, rewrite.Options{
		APIURL:      *apiURL,
		APIKey:      key,
		Model:       *model,
		Concurrency: *concurrency,
		Resume:      *resume,
	})
}

func input(fs *flag.FlagSet) (io.ReadCloser, error) {
	if fs.NArg() > 1 {
		return nil, errors.New("too many input files")
	}
	if fs.NArg() == 1 {
		return os.Open(fs.Arg(0))
	}
	return io.NopCloser(os.Stdin), nil
}

func output(path string, force bool) (io.Writer, func(), error) {
	if path == "" {
		return os.Stdout, func() {}, nil
	}
	if !force {
		if _, err := os.Stat(path); err == nil {
			return nil, nil, fmt.Errorf("output %s exists, use --force to overwrite", path)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, nil, err
	}
	return f, func() { f.Close() }, nil
}
