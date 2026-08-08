package scrape

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func run(t *testing.T, html string) string {
	t.Helper()
	var out bytes.Buffer
	if err := Run(strings.NewReader(html), &out); err != nil {
		t.Fatal(err)
	}
	return out.String()
}

func TestMinimal(t *testing.T) {
	if out := run(t, "<p>Hello world.</p>"); out != "Hello world." {
		t.Errorf("got %q, want %q", out, "Hello world.")
	}
}

func TestStdin(t *testing.T) {
	// Run takes any io.Reader; stdin is just another reader.
	if out := run(t, "<p>stdin path works.</p>"); out != "stdin path works." {
		t.Errorf("got %q, want %q", out, "stdin path works.")
	}
}

func TestPreDroppedBetweenParagraphs(t *testing.T) {
	out := run(t, "<p>First.</p><pre>code here</pre><p>Second.</p>")
	if out != "First.\n\nSecond." {
		t.Errorf("got %q, want %q", out, "First.\n\nSecond.")
	}
	if strings.Contains(out, "code here") {
		t.Errorf("pre content leaked: %q", out)
	}
}

func TestInlineCodeKept(t *testing.T) {
	out := run(t, "<p>Run <code>ls -la</code> in the shell.</p>")
	if out != "Run ls -la in the shell." {
		t.Errorf("got %q, want %q", out, "Run ls -la in the shell.")
	}
}

func TestImageAlt(t *testing.T) {
	out := run(t, `<img alt="Diagram" src="fig.png"><p>text</p><img alt="" src="spacer.png">`)
	if out != "Diagram\n\ntext" {
		t.Errorf("got %q, want %q", out, "Diagram\n\ntext")
	}
}

func TestMediaWikiFixture(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "mediawiki_fixture.html"))
	if err != nil {
		t.Fatal(err)
	}
	out := run(t, string(data))

	// Chrome and code are absent: TOC, edit links, footer, catlinks, scripts,
	// styles, citations, code blocks, URLs.
	for _, junk := range []string{
		"vector-toc", "mw-editsection", "catlinks", "printfooter",
		"footer", "Skip to content", "Category: Software", "Retrieved from",
		"console.log", "font-family", "git clone", "example.com",
		"[1]", "https://", "mw-content-text", "bodyContent",
	} {
		if strings.Contains(out, junk) {
			t.Errorf("output contains dropped chrome %q:\n%s", junk, out)
		}
	}

	// Prose paragraphs survive, blank-line separated.
	if !strings.Contains(out, "First paragraph with ragged whitespace.\n\nThe --force flag overwrites an existing file.") {
		t.Errorf("prose paragraphs not blank-line separated:\n%s", out)
	}

	// Heading on its own line.
	if !strings.Contains(out, "Installation\n\nFirst paragraph") {
		t.Errorf("heading not on its own line:\n%s", out)
	}

	// Note box, list items, image alt.
	for _, want := range []string{"Note: Some note text.", "List item one.", "List item two.", "Diagram"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q:\n%s", want, out)
		}
	}

	// Empty alt emits nothing; no code tags survive; whitespace collapsed.
	if strings.Contains(out, "spacer") {
		t.Errorf("empty alt leaked: %s", out)
	}
	if strings.Contains(out, "<code>") || strings.Contains(out, "<strong>") {
		t.Errorf("HTML tags leaked into output:\n%s", out)
	}
	if strings.Contains(out, "ragged whitespace.") && strings.Contains(out, "  ") {
		t.Errorf("whitespace not collapsed:\n%s", out)
	}

	// Entities decoded.
	if !strings.Contains(out, "Fish & chips cost © 5.") {
		t.Errorf("entities not decoded (want \"Fish & chips cost © 5.\"):\n%s", out)
	}
	if !strings.Contains(out, "2 < 3 is true.") {
		t.Errorf("&lt; not decoded (want \"2 < 3 is true.\"):\n%s", out)
	}

	// Inline code kept inside its sentence, anchor text kept without href.
	if !strings.Contains(out, "The --force flag overwrites an existing file.") {
		t.Errorf("inline code not kept in sentence:\n%s", out)
	}
	if !strings.Contains(out, "Visit the manual page.") {
		t.Errorf("anchor text not kept:\n%s", out)
	}

	// Citation superscript dropped.
	if !strings.Contains(out, "A sentence with a citation at the end.") {
		t.Errorf("reference superscript not dropped:\n%s", out)
	}
}

func TestScrapeArchWikiIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping ArchWiki integration in short mode")
	}
	const path = "/usr/share/doc/arch-wiki/html/en/Bash.html"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("ArchWiki docs not installed: %v", err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out bytes.Buffer
	if err := Run(f, &out); err != nil {
		t.Fatal(err)
	}
	s := out.String()
	for _, junk := range []string{"vector-toc", "mw-editsection", "catlinks", "Jump to content", "From ArchWiki"} {
		if strings.Contains(s, junk) {
			t.Errorf("output contains junk %q", junk)
		}
	}
	if !strings.Contains(s, "Bash is the default command-line shell on Arch Linux") {
		t.Errorf("Bash intro prose missing from output")
	}
}
