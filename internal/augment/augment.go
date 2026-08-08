// Package augment enriches extract rows with IPA and dictionary meanings.
package augment

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	"github.com/kafo-play-box/shisui/internal/jsonl"
	"github.com/kafo-play-box/shisui/internal/yomitan"
)

// InRow is one extract row to enrich.
type InRow struct {
	TargetWord string `json:"target_word"`
	RootWord   string `json:"root_word"`
	Phrase     string `json:"phrase"`
}

// OutRow is an InRow with IPA and meanings attached.
type OutRow struct {
	TargetWord string   `json:"target_word"`
	RootWord   string   `json:"root_word"`
	Phrase     string   `json:"phrase"`
	IPA        string   `json:"ipa"`
	Meanings   []string `json:"meanings"`
}

// Options controls the enrichment pipeline.
type Options struct {
	YomitanClient *yomitan.Client
	Concurrency   int
}

var retryDelay = time.Second

type result struct {
	index int
	entry yomitan.Entry
	err   error
}

// Run reads extract rows from r, looks each root word up in Yomitan, and
// writes the enriched rows to w in input order.
func Run(r io.Reader, w io.Writer, opts Options) error {
	if opts.YomitanClient == nil {
		return errors.New("augment: nil yomitan client")
	}
	if opts.Concurrency < 1 {
		opts.Concurrency = 1
	}
	rows, err := jsonl.ReadAll[InRow](r)
	if err != nil {
		return err
	}
	ctx := context.Background()
	results := make(chan result, len(rows))
	jobs := seqs(len(rows))
	var wg sync.WaitGroup
	for range opts.Concurrency {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				entry, err := fetch(ctx, opts.YomitanClient, rows[i].RootWord)
				if err != nil {
					results <- result{index: i, err: err}
					continue
				}
				results <- result{index: i, entry: entry}
			}
		}()
	}
	wg.Wait()
	close(results)
	out := make([]OutRow, len(rows))
	var fatal error
	for res := range results {
		if res.err == nil {
			row := rows[res.index]
			out[res.index] = OutRow{
				TargetWord: row.TargetWord,
				RootWord:   row.RootWord,
				Phrase:     row.Phrase,
				IPA:        res.entry.IPA,
				Meanings:   res.entry.Meanings,
			}
			continue
		}
		if yomitan.IsUnreachable(res.err) {
			if fatal == nil {
				fatal = res.err
			}
			continue
		}
		if fatal == nil {
			fmt.Fprintf(os.Stderr, "augment: warning: %s: %v\n", rows[res.index].RootWord, res.err)
			row := rows[res.index]
			out[res.index] = OutRow{TargetWord: row.TargetWord, RootWord: row.RootWord, Phrase: row.Phrase}
		}
	}
	if fatal != nil {
		return fatal
	}
	filtered := out[:0]
	skipped := 0
	for _, row := range out {
		if len(row.Meanings) == 0 {
			skipped++
			continue
		}
		filtered = append(filtered, row)
	}
	if skipped > 0 {
		fmt.Fprintf(os.Stderr, "augment: skipped %d row(s) with empty meanings\n", skipped)
	}
	return jsonl.WriteAll(w, filtered)
}

func seqs(n int) <-chan int {
	ch := make(chan int, n)
	go func() {
		defer close(ch)
		for i := 0; i < n; i++ {
			ch <- i
		}
	}()
	return ch
}

func fetch(ctx context.Context, client *yomitan.Client, word string) (yomitan.Entry, error) {
	entry, err := client.TermEntries(ctx, word)
	if err == nil || yomitan.IsUnreachable(err) {
		return entry, err
	}
	time.Sleep(retryDelay)
	return client.TermEntries(ctx, word)
}
