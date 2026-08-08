// Package yomitan queries a Yomitan server for term entries.
package yomitan

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

// Entry is the IPA and meanings extracted for a word.
type Entry struct {
	IPA      string
	Meanings []string
}

// Client talks to one Yomitan server.
type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewClient returns a Client for baseURL with the given per-request timeout.
func NewClient(baseURL string, timeout time.Duration) *Client {
	return &Client{
		BaseURL:    strings.TrimRight(baseURL, "/"),
		HTTPClient: &http.Client{Timeout: timeout},
	}
}

// TermEntriesResponse is the top-level /termEntries response.
type TermEntriesResponse struct {
	DictionaryEntries []DictionaryEntry `json:"dictionaryEntries"`
}

// DictionaryEntry is one dictionary's result for a term.
type DictionaryEntry struct {
	Headwords   []Headword   `json:"headwords"`
	Definitions []Definition `json:"definitions"`
}

// Headword is one reading of a term.
type Headword struct {
	Term    string `json:"term"`
	Reading string `json:"reading"`
}

// Definition holds one dictionary's raw entries for the term.
type Definition struct {
	Dictionary string        `json:"dictionary"`
	Entries    []interface{} `json:"entries"`
}

// IsUnreachable reports whether err means the Yomitan server could not be
// reached at all (as opposed to a per-request failure).
func IsUnreachable(err error) bool {
	var ue unreachableError
	return errors.As(err, &ue)
}

// TermEntries fetches IPA and meanings for word.
func (c *Client) TermEntries(ctx context.Context, word string) (Entry, error) {
	body, err := json.Marshal(map[string]string{"term": word})
	if err != nil {
		return Entry{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/termEntries", bytes.NewReader(body))
	if err != nil {
		return Entry{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return Entry{}, classifyConnErr(err, c.BaseURL)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Entry{}, fmt.Errorf("termEntries returned status %d", resp.StatusCode)
	}
	var t TermEntriesResponse
	if err := json.NewDecoder(resp.Body).Decode(&t); err != nil {
		return Entry{}, err
	}
	return extractEntry(t, word), nil
}

type unreachableError struct{ url string }

func (e unreachableError) Error() string {
	return "Yomitan server unreachable at " + e.url + ". Is Yomitan running?"
}

func classifyConnErr(err error, base string) error {
	var urlErr *url.Error
	if !errors.As(err, &urlErr) {
		return err
	}
	var opErr *net.OpError
	if errors.As(urlErr.Err, &opErr) && (opErr.Op == "dial" || opErr.Op == "lookup") {
		return unreachableError{url: base}
	}
	return err
}

func extractEntry(t TermEntriesResponse, word string) Entry {
	ipa := ""
	for _, de := range t.DictionaryEntries {
		for _, hw := range de.Headwords {
			if looksLikeIPA(hw.Reading) {
				ipa = hw.Reading
				break
			}
		}
		if ipa != "" {
			break
		}
	}
	var raw []string
	for _, de := range t.DictionaryEntries {
		for _, def := range de.Definitions {
			for _, ent := range def.Entries {
				switch v := ent.(type) {
				case string:
					raw = append(raw, extractPlain(v)...)
				case map[string]interface{}:
					if typ, _ := v["type"].(string); typ == "structured-content" {
						raw = append(raw, extractStructured(v)...)
					}
				}
			}
		}
	}
	if ipa == "" {
		ipa = espeak(context.Background(), word)
	}
	return Entry{IPA: ipa, Meanings: dedupCap(raw, 12)}
}

var espeak = espeakIPA

func espeakIPA(ctx context.Context, word string) string {
	cctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(cctx, "espeak-ng", "-q", "--ipa", word).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func looksLikeIPA(s string) bool {
	if strings.ContainsAny(s, "ˈˌæɑɒɔəɛɜɪʊʌʒŋðθʃ") {
		return true
	}
	return strings.Contains(s, "/") || strings.Contains(s, "[") || strings.Contains(s, "]")
}

var posWords = map[string]bool{
	"verb": true, "noun": true, "adjective": true, "adverb": true,
	"preposition": true, "conjunction": true, "pronoun": true,
	"determiner": true, "interjection": true, "prefix": true,
	"suffix": true, "article": true, "auxiliary verb": true,
	"phrasal verb": true, "idiom": true,
}

var numberedSenseRe = regexp.MustCompile(`^\d+[.)]\s*`)
var newOxfordSenseRe = regexp.MustCompile(`\n\d+\s`)

func extractPlain(s string) []string {
	var out []string
	if strings.Contains(firstLine(s), " | ") {
		return extractNewOxford(s)
	}
	for _, chunk := range strings.Split(s, "\n\t·") {
		lines := strings.Split(chunk, "\n")
		var starts []int
		for i, ln := range lines {
			if posWords[strings.ToLower(strings.TrimSpace(ln))] {
				starts = append(starts, i)
			}
		}
		if len(starts) == 0 {
			if m := numberedSenseRe.FindString(strings.TrimSpace(chunk)); m != "" {
				out = append(out, strings.TrimSpace(chunk)[len(m):])
			}
			continue
		}
		for k, si := range starts {
			end := len(lines)
			if k+1 < len(starts) {
				end = starts[k+1]
			}
			out = append(out, strings.Join(lines[si+1:end], " "))
		}
	}
	return out
}

func extractNewOxford(s string) []string {
	var out []string
	for _, sense := range newOxfordSenseRe.Split(s, -1)[1:] {
		if idx := firstColonCut(sense); idx >= 0 {
			sense = sense[:idx]
		}
		out = append(out, strings.TrimSpace(sense))
	}
	return out
}

func firstColonCut(s string) int {
	for i := 0; i < len(s); i++ {
		if s[i] == ':' && i+1 < len(s) && (s[i+1] == ' ' || s[i+1] == '\n' || s[i+1] == '\t') {
			return i
		}
	}
	return -1
}

func firstLine(s string) string {
	if idx := strings.IndexByte(s, '\n'); idx >= 0 {
		return s[:idx]
	}
	return s
}

var skipContent = map[string]bool{
	"preamble": true, "Grammar-content": true, "Etymology-content": true,
	"extra-info": true, "example-sentence": true, "example-sentence-a": true,
	"tags": true, "tag": true, "summary-entry": true, "backlink": true,
}

func extractStructured(entry map[string]interface{}) []string {
	var glosses []string
	walkStructured(entry, &glosses)
	return glosses
}

func walkStructured(node map[string]interface{}, out *[]string) {
	if skipNode(node) {
		return
	}
	if tagOf(node) == "ol" && dataContent(node) == "glosses" {
		for _, child := range nodeChildren(node) {
			if m, ok := child.(map[string]interface{}); ok {
				collectNodeText(m, out)
			}
		}
		return
	}
	for _, child := range nodeChildren(node) {
		if m, ok := child.(map[string]interface{}); ok {
			walkStructured(m, out)
		}
	}
}

func collectNodeText(node map[string]interface{}, out *[]string) {
	if skipNode(node) {
		return
	}
	for _, child := range nodeChildren(node) {
		switch v := child.(type) {
		case string:
			*out = append(*out, v)
		case map[string]interface{}:
			collectNodeText(v, out)
		}
	}
}

func skipNode(node map[string]interface{}) bool {
	if node["tag"] == "details" {
		return true
	}
	return skipContent[dataContent(node)]
}

func nodeChildren(node map[string]interface{}) []interface{} {
	switch v := node["content"].(type) {
	case []interface{}:
		return v
	case string:
		return []interface{}{v}
	}
	return nil
}

func dataContent(node map[string]interface{}) string {
	data, _ := node["data"].(map[string]interface{})
	dc, _ := data["content"].(string)
	return dc
}

func tagOf(node map[string]interface{}) string {
	t, _ := node["tag"].(string)
	return t
}

func dedupCap(glosses []string, cap int) []string {
	seen := make(map[string]bool, len(glosses))
	var out []string
	for _, g := range glosses {
		n := normalizeGloss(g)
		if n == "" || seen[n] {
			continue
		}
		seen[n] = true
		out = append(out, n)
		if len(out) >= cap {
			break
		}
	}
	return out
}

func normalizeGloss(s string) string {
	s = strings.ToLower(s)
	s = strings.Join(strings.Fields(s), " ")
	s = strings.TrimLeft(s, "-–—•·*")
	s = strings.TrimRight(s, ".,;:!?·•-–—\"'")
	return strings.TrimSpace(s)
}
