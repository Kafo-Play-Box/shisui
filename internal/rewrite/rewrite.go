// Package rewrite turns augment rows into LLM-written flashcard definitions.
package rewrite

import (
	"bufio"
	"bytes"
	"container/heap"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"sync"
	"text/template"
	"time"
	"unicode"

	_ "embed"

	"github.com/kafo-play-box/shisui/internal/jsonl"
)

// InRow is one augment row to rewrite.
type InRow struct {
	TargetWord string   `json:"target_word"`
	RootWord   string   `json:"root_word"`
	Phrase     string   `json:"phrase"`
	IPA        string   `json:"ipa"`
	Meanings   []string `json:"meanings"`
}

// OutRow is an InRow with an LLM definition and quality label.
type OutRow struct {
	TargetWord      string   `json:"target_word"`
	RootWord        string   `json:"root_word"`
	Phrase          string   `json:"phrase"`
	IPA             string   `json:"ipa"`
	Meanings        []string `json:"meanings"`
	SimpleMeaning   string   `json:"simple_meaning"`
	Quality         string   `json:"quality"`
	SelectedMeaning string   `json:"selected_meaning"`
	Key             string   `json:"_key"`
}

// Options controls the rewrite pipeline.
type Options struct {
	APIURL      string
	APIKey      string
	Model       string
	Concurrency int
	Resume      bool
}

var retryBackoff = []time.Duration{time.Second, 2 * time.Second}

type job struct {
	seq int
	row InRow
	key string
}

type result struct {
	seq    int
	row    OutRow
	failed bool
}

type resultHeap []result

func (h resultHeap) Len() int           { return len(h) }
func (h resultHeap) Less(i, j int) bool { return h[i].seq < h[j].seq }
func (h resultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *resultHeap) Push(x any)        { *h = append(*h, x.(result)) }
func (h *resultHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// Run reads augment rows from r, rewrites each via the LLM API, and appends
// the results to w in input order.
func Run(r io.Reader, w io.WriteSeeker, opts Options) error {
	if opts.APIURL == "" || opts.Model == "" {
		return errors.New("rewrite: api-url and model are required")
	}
	if opts.Concurrency < 1 {
		opts.Concurrency = 1
	}
	rows, err := jsonl.ReadAll[InRow](r)
	if err != nil {
		return err
	}
	var done map[string]bool
	if opts.Resume {
		done, err = scanKeys(w)
		if err != nil {
			return err
		}
	}
	var jobs []job
	for _, row := range rows {
		key := makeKey(row.TargetWord, row.Phrase)
		if done[key] {
			continue
		}
		jobs = append(jobs, job{seq: len(jobs), row: row, key: key})
	}
	if len(jobs) == 0 {
		return nil
	}
	ctx := context.Background()
	jobsCh := make(chan job)
	resultsCh := make(chan result)
	var wg sync.WaitGroup
	for range opts.Concurrency {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobsCh {
				resultsCh <- processJob(ctx, opts, j)
			}
		}()
	}
	go func() {
		for _, j := range jobs {
			jobsCh <- j
		}
		close(jobsCh)
		wg.Wait()
		close(resultsCh)
	}()
	return writeOrdered(resultsCh, w, len(jobs))
}

func scanKeys(w io.WriteSeeker) (map[string]bool, error) {
	rw, ok := w.(io.ReadSeeker)
	if !ok {
		return nil, errors.New("rewrite: resume requires a readable output file")
	}
	if _, err := rw.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	set := make(map[string]bool)
	sc := bufio.NewScanner(rw)
	for sc.Scan() {
		var row OutRow
		if err := json.Unmarshal(sc.Bytes(), &row); err != nil {
			fmt.Fprintf(os.Stderr, "rewrite: warning: skipping malformed resume row: %v\n", err)
			continue
		}
		if row.Key == "" {
			row.Key = makeKey(row.TargetWord, row.Phrase)
		}
		set[row.Key] = true
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if _, err := rw.Seek(0, io.SeekEnd); err != nil {
		return nil, err
	}
	return set, nil
}

func processJob(ctx context.Context, opts Options, j job) result {
	base := OutRow{
		TargetWord: j.row.TargetWord,
		RootWord:   j.row.RootWord,
		Phrase:     j.row.Phrase,
		IPA:        j.row.IPA,
		Meanings:   j.row.Meanings,
		Key:        j.key,
	}
	out, err := callLLM(ctx, opts, j.row, j.row.Meanings)
	if err != nil {
		fmt.Fprintf(os.Stderr, "rewrite: warning: %s: %v\n", j.row.TargetWord, err)
		base.Quality = "bad"
		return result{seq: j.seq, row: base, failed: true}
	}
	base.SimpleMeaning = out.SimpleMeaning
	base.Quality = out.Quality
	base.SelectedMeaning = out.SelectedMeaning
	return result{seq: j.seq, row: base, failed: false}
}

func writeOrdered(resultsCh <-chan result, w io.Writer, total int) error {
	enc := jsonl.NewEncoder[OutRow](w)
	var h resultHeap
	next := 0
	ok := 0
	failed := 0
	last := time.Now()
	report := func() {
		fmt.Fprintf(os.Stderr, "rewrite: processed %d/%d rows (%d%%), %d failed, %d ok\n", next, total, next*100/total, failed, ok)
	}
	for res := range resultsCh {
		heap.Push(&h, res)
		for h.Len() > 0 && h[0].seq == next {
			top := heap.Pop(&h).(result)
			next++
			if top.failed {
				failed++
			} else {
				ok++
			}
			if err := enc.Encode(top.row); err != nil {
				return err
			}
			if next%10 == 0 || time.Since(last) >= 5*time.Second {
				report()
				last = time.Now()
			}
		}
	}
	report()
	return nil
}

type llmResult struct {
	SimpleMeaning   string
	Quality         string
	SelectedMeaning string // raw model response, before matching
}

func callLLM(ctx context.Context, opts Options, row InRow, meanings []string) (llmResult, error) {
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			delay := retryBackoff[attempt-1]
			delay = time.Duration(float64(delay) * (0.8 + 0.4*rand.Float64()))
			time.Sleep(delay)
		}
		out, err := doCall(ctx, opts, row, meanings)
		if err == nil {
			return out, nil
		}
		lastErr = err
		if !retryable(err) {
			break
		}
	}
	return llmResult{}, lastErr
}

func doCall(ctx context.Context, opts Options, row InRow, meanings []string) (llmResult, error) {
	body, err := buildBody(opts, row)
	if err != nil {
		return llmResult{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(opts.APIURL, "/")+"/chat/completions", body)
	if err != nil {
		return llmResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if opts.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+opts.APIKey)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return llmResult{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return llmResult{}, apiError{code: resp.StatusCode}
	}
	var cr chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return llmResult{}, err
	}
	if len(cr.Choices) == 0 {
		return llmResult{}, errors.New("rewrite: empty choices")
	}
	return parseContent(cr.Choices[0].Message.Content, meanings)
}

type apiError struct{ code int }

func (e apiError) Error() string { return fmt.Sprintf("rewrite: api status %d", e.code) }

func retryable(err error) bool {
	var ae apiError
	if errors.As(err, &ae) {
		return ae.code == 429 || ae.code >= 500
	}
	return true
}

func parseContent(content string, meanings []string) (llmResult, error) {
	var parsed struct {
		SimpleMeaning   string `json:"simple_meaning"`
		Quality         string `json:"quality"`
		SelectedMeaning string `json:"selected_meaning"`
	}
	if err := json.Unmarshal([]byte(content), &parsed); err != nil {
		return llmResult{}, err
	}
	if strings.TrimSpace(parsed.SimpleMeaning) == "" {
		return llmResult{Quality: "bad", SelectedMeaning: ""}, nil
	}
	q := parsed.Quality
	if q != "good" && q != "medium" && q != "bad" {
		q = "bad"
	}
	verbatim, matched := matchMeaning(parsed.SelectedMeaning, meanings)
	if matched {
		return llmResult{SimpleMeaning: parsed.SimpleMeaning, Quality: q, SelectedMeaning: verbatim}, nil
	}
	if strings.TrimSpace(parsed.SelectedMeaning) != "" {
		// The model claimed a meaning we cannot verify against the list.
		q = "bad"
	}
	return llmResult{SimpleMeaning: parsed.SimpleMeaning, Quality: q}, nil
}

type chatRequest struct {
	Model          string            `json:"model"`
	Messages       []chatMessage     `json:"messages"`
	Temperature    float64           `json:"temperature"`
	MaxTokens      int               `json:"max_tokens"`
	ResponseFormat map[string]string `json:"response_format"`
	Thinking       map[string]string `json:"thinking,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

//go:embed prompts/system.txt
var systemPromptBytes []byte

//go:embed prompts/user.txt
var userPromptBytes []byte

var userPromptTmpl = template.Must(template.New("user").Parse(string(userPromptBytes)))

var systemPrompt = string(systemPromptBytes)

func buildBody(opts Options, row InRow) (io.Reader, error) {
	var ub bytes.Buffer
	if err := userPromptTmpl.Execute(&ub, row); err != nil {
		return nil, err
	}
	req := chatRequest{
		Model: opts.Model,
		Messages: []chatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: ub.String()},
		},
		Temperature:    0.3,
		MaxTokens:      150,
		ResponseFormat: map[string]string{"type": "json_object"},
	}
	// DeepSeek models are thinking-enabled by default and burn the whole
	// max_tokens budget on reasoning_content, returning empty content.
	if strings.HasPrefix(opts.Model, "deepseek") {
		req.Thinking = map[string]string{"type": "disabled"}
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	return bytes.NewReader(body), nil
}

func makeKey(target, phrase string) string {
	sum := sha256.Sum256([]byte(target + "\x00" + phrase))
	return hex.EncodeToString(sum[:])
}

// normalize lowercases s, drops everything that is not a letter, digit, or
// whitespace (punctuation, hyphens, parentheses), and collapses whitespace
// runs to single spaces. "To move, FAST!" -> "to move fast".
func normalize(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsSpace(r) {
			b.WriteRune(r)
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}

func tokenSet(s string) map[string]bool {
	fields := strings.Fields(s)
	set := make(map[string]bool, len(fields))
	for _, f := range fields {
		set[f] = true
	}
	return set
}

// matchThreshold is the minimum overlap ratio (model tokens ∩ meaning tokens,
// over model tokens) required to accept the model's selected_meaning as a
// verbatim match of a provided meaning. Small models paraphrase instead of
// copying, so the match is done tool-side on normalized tokens.
const matchThreshold = 0.80

// matchMeaning reports whether the model's raw selected_meaning matches one of
// the provided meanings by normalized token overlap, returning the verbatim
// list entry. Comparison is strict; ties keep the earlier list entry.
func matchMeaning(raw string, meanings []string) (string, bool) {
	if strings.TrimSpace(raw) == "" || len(meanings) == 0 {
		return "", false
	}
	normRaw := normalize(raw)
	if normRaw == "" {
		return "", false
	}
	modelSet := tokenSet(normRaw)
	bestIdx := -1
	bestRatio := 0.0
	for i, m := range meanings {
		meaningSet := tokenSet(normalize(m))
		inter := 0
		for tok := range modelSet {
			if meaningSet[tok] {
				inter++
			}
		}
		ratio := float64(inter) / float64(len(modelSet))
		if ratio > bestRatio {
			bestRatio = ratio
			bestIdx = i
		}
	}
	if bestIdx >= 0 && bestRatio >= matchThreshold {
		return meanings[bestIdx], true
	}
	return "", false
}
