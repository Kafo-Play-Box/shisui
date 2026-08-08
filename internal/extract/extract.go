// Package extract turns plaintext into lemma-targeted phrase rows.
package extract

import (
	"errors"
	"io"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/kafo-play-box/shisui/internal/jsonl"
	"github.com/kafo-play-box/shisui/internal/lemma"
)

// defaultMinTargetLen is the minimum rune length a target candidate must have
// when Options.MinTargetLen is unset.
const defaultMinTargetLen = 3

// Row is one extracted phrase targeting a word.
type Row struct {
	TargetWord string `json:"target_word"`
	RootWord   string `json:"root_word"`
	Phrase     string `json:"phrase"`
}

// Options controls phrase generation.
type Options struct {
	LemmaDB  *lemma.DB
	MinWords int
	MaxWords int
	// MinTargetLen is the minimum length in runes a token must have to be
	// eligible as a target. Shorter tokens stay in the phrase for context
	// but can never be the target. Values below 1 fall back to
	// defaultMinTargetLen.
	MinTargetLen int
}

var (
	urlRe       = regexp.MustCompile(`https?://\S+|www\.\S+`)
	figRe       = regexp.MustCompile(`(?i)Figure\s*\d+\s*:?\s*`)
	hashRe      = regexp.MustCompile(`(?m)^#\w+\s*`)
	wsRe        = regexp.MustCompile(`[ \t]+`)
	nlRe        = regexp.MustCompile(`\n{3,}`)
	domainRe    = regexp.MustCompile(`(?i)\b[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,}(?:/[^\s]*)?\b`)
	uuidRe      = regexp.MustCompile(`(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b`)
	doiRe       = regexp.MustCompile(`\b10\.\d{4,}/[A-Za-z0-9._/-]+\b`)
	codeLineRe  = regexp.MustCompile(`(?m)^[ \t]*(?:import |from |def |if __name__|print\(|return |# )`)
	codeTokenRe = regexp.MustCompile(` = \[|= \{|\bHYDE_PROMPT\b|\bPII_PATTERNS\b|client\.messages\.create\(|np\.argsort|\.tolist\(\)| @ |logger\.`)
	publishedRe = regexp.MustCompile(`(?m)^Published:`)
)

func clean(text string) string {
	s := urlRe.ReplaceAllString(text, "")
	s = domainRe.ReplaceAllString(s, "")
	s = uuidRe.ReplaceAllString(s, "")
	s = doiRe.ReplaceAllString(s, "")
	s = figRe.ReplaceAllString(s, "")
	s = hashRe.ReplaceAllString(s, "")
	s = wsRe.ReplaceAllString(s, " ")
	s = nlRe.ReplaceAllString(s, "\n\n")
	return strings.TrimSpace(s)
}

// filterBoilerplate drops scraped-web junk paragraphs (code, nav lists,
// too-short filler) before sentence splitting. A paragraph with fewer than
// minWords can never yield a row, so dropping it is lossless.
func filterBoilerplate(text string, minWords int) string {
	var keep []string
	for _, p := range strings.Split(text, "\n\n") {
		p = strings.TrimSpace(p)
		if p == "" || isCodeParagraph(p) || isNavList(p) || publishedRe.MatchString(p) || wordCount(p) < minWords {
			continue
		}
		p = collapseDupLines(p)
		if hasDupLine(p) {
			continue
		}
		keep = append(keep, p)
	}
	return strings.Join(keep, "\n\n")
}

func isCodeParagraph(p string) bool {
	return codeLineRe.MatchString(p) || codeTokenRe.MatchString(p)
}

func isNavList(p string) bool {
	lines := strings.Split(p, "\n")
	if len(lines) < 2 {
		return false
	}
	for _, l := range lines {
		if wordCount(l) > 5 || strings.ContainsAny(l, ".!?;:…") {
			return false
		}
	}
	return true
}

// collapseDupLines removes immediately repeated lines
// ("Sameer Shukla\nSameer Shukla" -> "Sameer Shukla").
func collapseDupLines(p string) string {
	lines := strings.Split(p, "\n")
	out := lines[:1]
	for _, l := range lines[1:] {
		if l != out[len(out)-1] {
			out = append(out, l)
		}
	}
	return strings.Join(out, "\n")
}

// hasDupLine reports whether any line of p appears more than once. Scraped
// article headers echo the title around the byline ("title\nbyline\ntitle");
// real prose never repeats a full line.
func hasDupLine(p string) bool {
	seen := make(map[string]bool, wordCount(p)/5+2)
	for _, l := range strings.Split(p, "\n") {
		if seen[l] {
			return true
		}
		seen[l] = true
	}
	return false
}

func wordCount(s string) int {
	return len(strings.Fields(s))
}

// Run extracts rows from the plaintext on r and writes JSONL to w.
func Run(r io.Reader, w io.Writer, opts Options) error {
	if opts.LemmaDB == nil {
		return errors.New("extract: nil lemma db")
	}
	if opts.MinWords < 1 || opts.MaxWords < opts.MinWords {
		return errors.New("extract: min-words must be at least 1 and max-words at least min-words")
	}
	if opts.MinTargetLen < 1 {
		opts.MinTargetLen = defaultMinTargetLen
	}
	text, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	enc := jsonl.NewEncoder[Row](w)
	seen := make(map[string]bool)
	for _, sent := range splitSentences(filterBoilerplate(clean(string(text)), opts.MinWords)) {
		toks := tokenize(sent)
		if len(toks) < opts.MinWords {
			continue
		}
		bestIdx, bestRoot, bestFreq := -1, "", 0
		for i, tok := range toks {
			if utf8.RuneCountInString(tok.word) < opts.MinTargetLen {
				continue
			}
			root, freq, ok := opts.LemmaDB.Lookup(tok.word)
			if !ok {
				continue
			}
			if bestIdx == -1 || freq < bestFreq {
				bestIdx, bestRoot, bestFreq = i, root, freq
			}
		}
		if bestIdx == -1 {
			continue
		}
		phrase := phraseFor(sent, toks, bestIdx, opts.MaxWords)
		if seen[phrase] {
			continue
		}
		seen[phrase] = true
		if err := enc.Encode(Row{TargetWord: toks[bestIdx].word, RootWord: bestRoot, Phrase: phrase}); err != nil {
			return err
		}
	}
	return nil
}

type token struct {
	word  string
	start int
	end   int
}

func splitSentences(text string) []string {
	var out []string
	start := 0
	for i := 0; i < len(text); i++ {
		c := text[i]
		if c != '.' && c != '!' && c != '?' {
			continue
		}
		j := i + 1
		for j < len(text) && (text[j] == '"' || text[j] == '\'' || text[j] == ' ' || text[j] == '\t' || text[j] == '\n' || text[j] == '\r') {
			j++
		}
		if j < len(text) && text[j] >= 'A' && text[j] <= 'Z' {
			out = append(out, text[start:j])
			start = j
			i = j - 1
		}
	}
	if start < len(text) {
		out = append(out, text[start:])
	}
	return out
}

func tokenize(sent string) []token {
	var toks []token
	i := 0
	for i < len(sent) {
		for i < len(sent) && isSpace(sent[i]) {
			i++
		}
		if i >= len(sent) {
			break
		}
		j := i
		for j < len(sent) && !isSpace(sent[j]) {
			j++
		}
		raw := sent[i:j]
		k := i
		l := j
		for k < l && isPunct(raw[k-i]) {
			k++
		}
		for l > k && isPunct(raw[l-1-i]) {
			l--
		}
		if k < l {
			toks = append(toks, token{word: sent[k:l], start: i, end: j})
		}
		i = j
	}
	return toks
}

func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

func isPunct(c byte) bool {
	switch c {
	case '.', ',', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']':
		return true
	}
	return false
}

func phraseFor(sent string, toks []token, i, maxWords int) string {
	start := 0
	end := len(toks)
	if len(toks) > maxWords {
		start = i - maxWords/2
		if start < 0 {
			start = 0
		}
		end = start + maxWords
		if end > len(toks) {
			end = len(toks)
			start = end - maxWords
		}
	}
	s := sent[toks[start].start:toks[end-1].end]
	return strings.TrimSpace(strings.Join(strings.Fields(s), " "))
}
