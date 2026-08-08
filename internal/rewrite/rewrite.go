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
	TargetWord    string   `json:"target_word"`
	RootWord      string   `json:"root_word"`
	Phrase        string   `json:"phrase"`
	IPA           string   `json:"ipa"`
	Meanings      []string `json:"meanings"`
	SimpleMeaning string   `json:"simple_meaning"`
	Quality       string   `json:"quality"`
	Key           string   `json:"_key"`
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
	out, err := callLLM(ctx, opts, j.row)
	if err != nil {
		fmt.Fprintf(os.Stderr, "rewrite: warning: %s: %v\n", j.row.TargetWord, err)
		base.Quality = "bad"
		return result{seq: j.seq, row: base, failed: true}
	}
	base.SimpleMeaning = out.SimpleMeaning
	base.Quality = out.Quality
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
	SimpleMeaning string
	Quality       string
}

func callLLM(ctx context.Context, opts Options, row InRow) (llmResult, error) {
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			delay := retryBackoff[attempt-1]
			delay = time.Duration(float64(delay) * (0.8 + 0.4*rand.Float64()))
			time.Sleep(delay)
		}
		out, err := doCall(ctx, opts, row)
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

func doCall(ctx context.Context, opts Options, row InRow) (llmResult, error) {
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
	return parseContent(cr.Choices[0].Message.Content)
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

func parseContent(content string) (llmResult, error) {
	var parsed struct {
		SimpleMeaning string `json:"simple_meaning"`
		Quality       string `json:"quality"`
	}
	if err := json.Unmarshal([]byte(content), &parsed); err != nil {
		return llmResult{}, err
	}
	if strings.TrimSpace(parsed.SimpleMeaning) == "" {
		return llmResult{Quality: "bad"}, nil
	}
	q := parsed.Quality
	if q != "good" && q != "medium" && q != "bad" {
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

var userPromptTmpl = template.Must(template.New("user").Parse(`Word: {{.TargetWord}}
Phrase: "{{.Phrase}}"
Dictionary meanings:
{{range .Meanings}}- {{.}}
{{end}}Return JSON with exactly these fields:
{
  "simple_meaning": "your short, simple definition",
  "quality": "good" | "medium" | "bad"
}

Quality guide:
- "good": phrase context clearly matches one dictionary meaning, definition is accurate and simple.
- "medium": context somewhat matches but the meaning could apply to other words too, or definition is a bit vague.
- "bad": phrase lacks enough context to pick the right meaning from the dictionary list.
`))

const systemPrompt = `You are an ESL flashcard definition writer. Given a word used in a phrase and its dictionary meanings, write a short, clear English definition that helps an ESL learner understand the word's usage in that specific context.

Rules:
- Use ONLY the provided dictionary meanings. Do not invent new meanings.
- Write in Simple English (CEFR A2-B1 vocabulary).
- Keep the definition under 25 words.
- Be unambiguous. If the phrase doesn't give enough context to pick the right meaning, say so in the quality field.
- Return ONLY valid JSON. No other text.`

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
