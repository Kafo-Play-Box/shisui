package extract

import (
	"bytes"
	"os"
	"regexp"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/kafo-play-box/shisui/internal/jsonl"
	"github.com/kafo-play-box/shisui/internal/lemma"
)

const testLemmaData = `; tiny lemma db for tests
be/4109826 -> is,was,were,are,been
do/535646 -> did,does,done,doing
run/44715 -> running,ran,runs
it/1213224 -> its
k/10 -> k
cs/20 -> cs
bank/100 -> banks
cat/5400 -> cats
fox/5000 -> foxes
dog/8000 -> dogs
jump/2500 -> jumps
river/3000 -> rivers
distribution/3000 -> distributions
documentation/4000 -> documentations
package/100 -> packages
version/3500 -> versions
software/2500 -> softwares
stable/5000 -> stables,stably
system/1500 -> systems
manual/4500 -> manuals
community/3000 -> communities
model/1000 -> models
command/3000 -> commands
process/2000 -> processes
guide/4000 -> guides
installation/3500 -> installations
'hood -> 'hoods
quay -> quays
`

func testDB(t *testing.T) *lemma.DB {
	t.Helper()
	db, err := lemma.Parse(strings.NewReader(testLemmaData))
	if err != nil {
		t.Fatal(err)
	}
	return db
}

func runExtract(t *testing.T, text string, db *lemma.DB, min, max int) []Row {
	t.Helper()
	return runExtractOpts(t, text, Options{LemmaDB: db, MinWords: min, MaxWords: max})
}

func runExtractOpts(t *testing.T, text string, opts Options) []Row {
	t.Helper()
	var buf bytes.Buffer
	if err := Run(strings.NewReader(text), &buf, opts); err != nil {
		t.Fatal(err)
	}
	rows, err := jsonl.ReadAll[Row](&buf)
	if err != nil {
		t.Fatal(err)
	}
	return rows
}

func loremText(t *testing.T) string {
	t.Helper()
	data, err := os.ReadFile("../../testdata/lorem.txt")
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

const bankSixWord = "The river bank was steep here."
const bankEighteenWord = "The old bank stood beside the wide river where the fishermen gathered every morning long before the sunrise."

func TestMinWordsThreshold(t *testing.T) {
	rows := runExtract(t, loremText(t), testDB(t), 10, 80)
	for _, row := range rows {
		if row.Phrase == bankSixWord {
			t.Errorf("6-word sentence survived min-words=10: %+v", row)
		}
	}
}

func TestSufficientContext(t *testing.T) {
	rows := runExtract(t, loremText(t), testDB(t), 10, 80)
	found := 0
	for _, row := range rows {
		if row.Phrase == bankEighteenWord {
			found++
			if row.TargetWord != "bank" || row.RootWord != "bank" {
				t.Errorf("bank row = %+v, want target and root bank", row)
			}
		}
	}
	if found != 1 {
		t.Fatalf("found %d rows with the 18-word phrase, want 1", found)
	}
}

func TestRunningLemma(t *testing.T) {
	rows := runExtract(t, loremText(t), testDB(t), 10, 80)
	for _, row := range rows {
		if row.TargetWord == "running" {
			if row.RootWord != "run" {
				t.Errorf("running -> root %q, want run", row.RootWord)
			}
			if got := wordCount(row.Phrase); got != 12 {
				t.Errorf("running phrase has %d words, want the full 12-word sentence", got)
			}
		}
	}
}

func TestLongSentenceTruncation(t *testing.T) {
	rows := runExtract(t, loremText(t), testDB(t), 10, 40)
	truncated := false
	for _, row := range rows {
		words := wordCount(row.Phrase)
		if words > 40 {
			t.Errorf("phrase exceeds max-words=40: %q (%d words)", row.Phrase, words)
		}
		if words == 40 {
			truncated = true
		}
	}
	if !truncated {
		t.Fatal("no phrase hit the 40-word limit; truncation did not run")
	}
	for _, row := range rows {
		if row.RootWord != "package" {
			continue
		}
		idx := wordIndex(row.Phrase, "package")
		if idx < 15 || idx > 25 {
			t.Errorf("package target at phrase index %d, want near the middle (15-25)", idx)
		}
	}
}

func TestDedup(t *testing.T) {
	rows := runExtract(t, loremText(t), testDB(t), 6, 80)
	seen := make(map[string]bool)
	for _, row := range rows {
		if seen[row.Phrase] {
			t.Errorf("duplicate phrase row: %+v", row)
		}
		seen[row.Phrase] = true
	}
}

func TestDedupInline(t *testing.T) {
	const text = "The cat ran to the bank.\n\nThe cat ran to the bank.\n"
	rows := runExtract(t, text, testDB(t), 6, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1 (identical sentences dedup to one)", len(rows))
	}
	if rows[0].Phrase != "The cat ran to the bank." {
		t.Errorf("unexpected phrase %q", rows[0].Phrase)
	}
}

func TestLemmaMatching(t *testing.T) {
	const text = "The cats are running to the river."
	rows := runExtract(t, text, testDB(t), 1, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1 (river is least common)", len(rows))
	}
	if rows[0].TargetWord != "river" || rows[0].RootWord != "river" {
		t.Errorf("row = %+v, want river over cats/are/running", rows[0])
	}
}

func TestContractions(t *testing.T) {
	const text = "I don't know it's wrong."
	rows := runExtract(t, text, testDB(t), 3, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	if rows[0].TargetWord != "don't" || rows[0].RootWord != "do" {
		t.Errorf("row = %+v, want don't -> do (freq 535646) over it's -> it (freq 1213224)", rows[0])
	}
}

func TestStopwordFreeTargets(t *testing.T) {
	const text = "The river bank and the cat ran."
	rows := runExtract(t, text, testDB(t), 1, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	if rows[0].TargetWord != "bank" || rows[0].RootWord != "bank" {
		t.Errorf("row = %+v, want bank over river/cat/ran", rows[0])
	}
}

func TestPhraseTrimmedAndFlattened(t *testing.T) {
	const text = "  The  cat\nran to the bank.  "
	rows := runExtract(t, text, testDB(t), 1, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	for _, row := range rows {
		if row.Phrase != strings.TrimSpace(row.Phrase) {
			t.Errorf("phrase not trimmed: %q", row.Phrase)
		}
		if strings.ContainsAny(row.Phrase, "\n\t") {
			t.Errorf("phrase contains raw whitespace: %q", row.Phrase)
		}
	}
}

func TestLeastCommonTarget_FrequencyOrdering(t *testing.T) {
	cases := []struct {
		text   string
		target string
		root   string
	}{
		{"The cats ran to the bank by the river.", "bank", "bank"},
		{"The cats are running to the river.", "river", "river"},
		{"The quays near the river bank.", "quays", "quay"},
	}
	for _, c := range cases {
		rows := runExtract(t, c.text, testDB(t), 1, 80)
		if len(rows) != 1 {
			t.Fatalf("%q: got %d rows, want 1", c.text, len(rows))
		}
		if rows[0].TargetWord != c.target || rows[0].RootWord != c.root {
			t.Errorf("%q: row = %+v, want target %q root %q", c.text, rows[0], c.target, c.root)
		}
	}
}

func TestLeastCommonTarget_TieRule(t *testing.T) {
	const text = "The bank and the package arrived."
	rows := runExtract(t, text, testDB(t), 1, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	if rows[0].TargetWord != "bank" {
		t.Errorf("row = %+v, want first-encountered bank on tie with package (freq 100)", rows[0])
	}
}

func TestOneRowPerSentence(t *testing.T) {
	const text = "The cats and the dogs and the foxes ran beside the river."
	rows := runExtract(t, text, testDB(t), 1, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1 (one row per sentence)", len(rows))
	}
	if rows[0].TargetWord != "river" {
		t.Errorf("row = %+v, want river (least common of cats/dogs/foxes/ran/river)", rows[0])
	}
}

func TestPhraseOnlyDedup(t *testing.T) {
	const text = "The cat ran to the bank.\nThe cat ran to the bank.\nA fox crossed the river bank.\n"
	rows := runExtract(t, text, testDB(t), 6, 80)
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want 2 (duplicate phrase deduped)", len(rows))
	}
	seen := make(map[string]bool)
	for _, row := range rows {
		if seen[row.Phrase] {
			t.Errorf("duplicate phrase %q", row.Phrase)
		}
		seen[row.Phrase] = true
	}
}

func TestMissingFreqIsRarest(t *testing.T) {
	const text = "The quays by the river bank."
	rows := runExtract(t, text, testDB(t), 1, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	if rows[0].TargetWord != "quays" || rows[0].RootWord != "quay" {
		t.Errorf("row = %+v, want quay (freq 0) over river/bank", rows[0])
	}
}

func TestMinTargetLen_ShortOnlySentenceDropped(t *testing.T) {
	// k, cs and it are the only lemmatizable words, all shorter than the
	// default MinTargetLen of 3: the sentence must yield no row.
	const text = "The k cs it."
	rows := runExtract(t, text, testDB(t), 1, 80)
	if len(rows) != 0 {
		t.Fatalf("got %d rows, want 0 (every lemmatizable word shorter than 3 runes)", len(rows))
	}
}

func TestMinTargetLen_ExplicitOneRestoresShortTargets(t *testing.T) {
	// With the filter at 1, the short words are eligible again and the
	// least-common one (k, freq 10) is chosen as before.
	const text = "The k cs it."
	rows := runExtractOpts(t, text, Options{LemmaDB: testDB(t), MinWords: 1, MaxWords: 80, MinTargetLen: 1})
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	if rows[0].TargetWord != "k" || rows[0].RootWord != "k" {
		t.Errorf("row = %+v, want k (freq 10) over cs (freq 20) and it", rows[0])
	}
}

func TestMinTargetLen_MixNeverPicksShortTarget(t *testing.T) {
	// k has the lowest frequency (10) but is shorter than the threshold, so
	// bank (100) must win over k and ran (run, 44715). No row may ever carry
	// a target shorter than MinTargetLen.
	const text = "The k ran to the bank."
	rows := runExtractOpts(t, text, Options{LemmaDB: testDB(t), MinWords: 1, MaxWords: 80, MinTargetLen: 3})
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	if rows[0].TargetWord != "bank" || rows[0].RootWord != "bank" {
		t.Errorf("row = %+v, want bank (k filtered out despite freq 10)", rows[0])
	}
	for _, row := range rows {
		if utf8.RuneCountInString(row.TargetWord) < 3 {
			t.Errorf("target %q shorter than MinTargetLen 3", row.TargetWord)
		}
	}
}

func TestCleaningURLs(t *testing.T) {
	const text = "The cat saw https://example.com/foo?x=1 the bank.\nSee www.example.com today."
	rows := runExtract(t, text, testDB(t), 5, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	if rows[0].Phrase != "The cat saw the bank." {
		t.Errorf("phrase = %q, want %q", rows[0].Phrase, "The cat saw the bank.")
	}
	if strings.Contains(rows[0].Phrase, "http") || strings.Contains(rows[0].Phrase, "www") {
		t.Errorf("phrase still contains a URL: %q", rows[0].Phrase)
	}
}

func TestCleaningFigure(t *testing.T) {
	const text = "Figure 1: The cat ran to the bank.\nfigure 3: The dog barked loudly.\n"
	rows := runExtract(t, text, testDB(t), 4, 80)
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want 2 (both caption prefixes stripped)", len(rows))
	}
	want := []string{"The cat ran to the bank.", "The dog barked loudly."}
	for i, row := range rows {
		if row.Phrase != want[i] {
			t.Errorf("phrase %d = %q, want %q", i, row.Phrase, want[i])
		}
		if strings.Contains(row.Phrase, "Figure") || strings.Contains(row.Phrase, "figure") {
			t.Errorf("phrase still contains a figure caption: %q", row.Phrase)
		}
	}
}

func TestCleaningHashtags(t *testing.T) {
	const text = "#RAG\nWhat Is HyDE?\nThe cat ran to the bank.\n"
	rows := runExtract(t, text, testDB(t), 5, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	if rows[0].TargetWord != "bank" {
		t.Errorf("row = %+v, want bank target", rows[0])
	}
	if strings.Contains(rows[0].Phrase, "#") {
		t.Errorf("phrase contains a hashtag: %q", rows[0].Phrase)
	}
}

func TestCleaningHyphens(t *testing.T) {
	const text = "The cat ran past the well-known river.\n"
	rows := runExtract(t, text, testDB(t), 6, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	if rows[0].TargetWord != "river" {
		t.Errorf("row = %+v, want river target", rows[0])
	}
	if !strings.Contains(rows[0].Phrase, "well-known") {
		t.Errorf("hyphenated word broken in phrase %q", rows[0].Phrase)
	}
}

func TestCleaningDomains(t *testing.T) {
	const text = "The cat saw freeCodeCamp.org and console.anthropic.com today."
	rows := runExtract(t, text, testDB(t), 1, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	if rows[0].Phrase != "The cat saw and today." {
		t.Errorf("phrase = %q, want %q", rows[0].Phrase, "The cat saw and today.")
	}
	if strings.Contains(rows[0].Phrase, "freeCodeCamp") || strings.Contains(rows[0].Phrase, "anthropic") || strings.Contains(rows[0].Phrase, "console") {
		t.Errorf("phrase still contains a domain: %q", rows[0].Phrase)
	}
}

func TestCleaningDomains_NegativesPreserved(t *testing.T) {
	// T. rex, U.S., e.g. are not domains (a dot needs a TLD right after it);
	// the DOI 10.64628/AA.tafck7dd3 is stripped entirely by doiRe, not as a
	// domain, and must leave no remnant behind. Asserted on clean() because
	// splitSentences splits "U.S." at "U."/"S." regardless of cleaning.
	got := clean("T. rex in the U.S. via e.g., but 10.64628/AA.tafck7dd3 vanished.")
	for _, frag := range []string{"T. rex", "U.S.", "e.g."} {
		if !strings.Contains(got, frag) {
			t.Errorf("clean() lost prose %q: %q", frag, got)
		}
	}
	if strings.Contains(got, "10.64628") || strings.Contains(got, "tafck7dd3") {
		t.Errorf("clean() left a DOI remnant: %q", got)
	}
}

func TestCleaningUUID(t *testing.T) {
	const text = "The cat lost id 123e4567-e89b-12d3-a456-426614174000 in the bank."
	rows := runExtract(t, text, testDB(t), 1, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	if rows[0].Phrase != "The cat lost id in the bank." {
		t.Errorf("phrase = %q, want %q", rows[0].Phrase, "The cat lost id in the bank.")
	}
	if strings.Contains(rows[0].Phrase, "123e4567") || strings.Contains(rows[0].Phrase, "-") {
		t.Errorf("phrase still contains a UUID remnant: %q", rows[0].Phrase)
	}
}

func TestCleaningDOI(t *testing.T) {
	const text = "The cat read 10.64628/AA.tafck7dd3 and ran."
	rows := runExtract(t, text, testDB(t), 1, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	if rows[0].Phrase != "The cat read and ran." {
		t.Errorf("phrase = %q, want %q", rows[0].Phrase, "The cat read and ran.")
	}
	if strings.Contains(rows[0].Phrase, "10.64628") || strings.Contains(rows[0].Phrase, "tafck7dd3") {
		t.Errorf("phrase still contains a DOI remnant: %q", rows[0].Phrase)
	}
}

func TestCleaningCodeParagraph(t *testing.T) {
	const text = "The cat ran to the bank.\n\n" +
		"import numpy as np\nfrom sentence_transformers import SentenceTransformer\n\n" +
		"collection = [\n    \"AWS Lambda reclaims idle execution environments.\",\n]\n\n" +
		"def retrieve(query):\n    return query\n\n" +
		"The dog barked very loudly.\n"
	rows := runExtract(t, text, testDB(t), 5, 80)
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want 2 (code block yields none)", len(rows))
	}
	want := []string{"The cat ran to the bank.", "The dog barked very loudly."}
	for i, row := range rows {
		if row.Phrase != want[i] {
			t.Errorf("phrase %d = %q, want %q", i, row.Phrase, want[i])
		}
		for _, junk := range []string{"import", "collection", "def", "numpy", "retrieve"} {
			if strings.Contains(row.Phrase, junk) {
				t.Errorf("phrase %d %q contains code junk %q", i, row.Phrase, junk)
			}
		}
	}
}

func TestCleaningNavList(t *testing.T) {
	const text = "Menu\nDonate\nJuly 22, 2026\n\nThe cat ran to the bank."
	rows := runExtract(t, text, testDB(t), 5, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1 (nav block yields none)", len(rows))
	}
	if rows[0].Phrase != "The cat ran to the bank." {
		t.Errorf("phrase = %q, want %q", rows[0].Phrase, "The cat ran to the bank.")
	}
}

func TestCleaningRepeatedByline(t *testing.T) {
	const text = "Sameer Shukla\nSameer Shukla\n\nThe cat ran to the bank."
	rows := runExtract(t, text, testDB(t), 5, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1 (repeated byline yields none)", len(rows))
	}
	if rows[0].Phrase != "The cat ran to the bank." {
		t.Errorf("phrase = %q, want %q", rows[0].Phrase, "The cat ran to the bank.")
	}
}

func TestCleaningTitleBylineTitle(t *testing.T) {
	// Scraped article headers echo the title around the byline. The repeated
	// title line marks the whole paragraph as boilerplate; no row may carry
	// the byline.
	const text = "What Is HyDE? How to Improve RAG with Hypothetical Documents\n" +
		"Sameer Shukla\nSameer Shukla\n" +
		"What Is HyDE? How to Improve RAG with Hypothetical Documents\n\n" +
		"The cat ran to the bank."
	rows := runExtract(t, text, testDB(t), 5, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1 (title/byline echo yields none)", len(rows))
	}
	if rows[0].Phrase != "The cat ran to the bank." {
		t.Errorf("phrase = %q, want %q", rows[0].Phrase, "The cat ran to the bank.")
	}
	if strings.Contains(rows[0].Phrase, "Sameer") || strings.Contains(rows[0].Phrase, "Shukla") {
		t.Errorf("phrase %q still contains the byline", rows[0].Phrase)
	}
}

func TestCleaningProseSurvives(t *testing.T) {
	const text = "Instead of asking an LLM to answer entirely from its training data, a RAG system retrieves relevant information from an external knowledge base and provides that information to the model as context."
	rows := runExtract(t, text, testDB(t), 5, 80)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	if rows[0].TargetWord != "model" || rows[0].RootWord != "model" {
		t.Errorf("row = %+v, want model target", rows[0])
	}
	if !strings.Contains(rows[0].Phrase, "Instead of asking an LLM to answer entirely from its training data") {
		t.Errorf("phrase lost the opening clause: %q", rows[0].Phrase)
	}
}

func TestCorpusIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping corpus integration in -short mode")
	}
	text, err := os.ReadFile("../../test-corpus.txt")
	if err != nil {
		if os.IsNotExist(err) {
			t.Skip("test-corpus.txt not present (gitignored); skipping")
		}
		t.Fatal(err)
	}
	lemmaData, err := os.ReadFile("../../lemma.en.txt")
	if err != nil {
		t.Fatal(err)
	}
	db, err := lemma.Parse(bytes.NewReader(lemmaData))
	if err != nil {
		t.Fatal(err)
	}
	rows := runExtract(t, string(text), db, 10, 80)
	// 1284 rows at baseline; phrase cleaning (code blocks, nav lists,
	// article headers) drops the scraped junk. 1000 leaves a comfortable
	// margin below the measured 1230 while still catching a broken filter.
	if len(rows) < 1000 {
		t.Fatalf("got %d rows, want at least 1000", len(rows))
	}
	noise := regexp.MustCompile(`https?://|www\.|Figure[ ]*[0-9]|#[A-Z]|import |collection = \[|Sameer Shukla|Menu|Donate|Skip to content|Published:|Table of Contents|freeCodeCamp\.org|console\.anthropic\.com|10\.64628|[0-9a-f]{8}-[0-9a-f]{4}`)
	seen := make(map[string]bool)
	for _, row := range rows {
		if noise.MatchString(row.Phrase) {
			t.Errorf("noisy phrase: %q", row.Phrase)
		}
		if row.TargetWord == "" || row.RootWord == "" {
			t.Errorf("empty target/root: %+v", row)
		}
		if seen[row.Phrase] {
			t.Errorf("duplicate phrase: %q", row.Phrase)
		}
		seen[row.Phrase] = true
	}
}

func wordIndex(phrase, word string) int {
	for i, w := range strings.Fields(phrase) {
		if strings.Trim(w, ".,!?;:\"'()[]") == word {
			return i
		}
	}
	return -1
}
