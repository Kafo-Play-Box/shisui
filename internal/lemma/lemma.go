// Package lemma parses and queries the lemma database.
package lemma

import (
	"bufio"
	"io"
	"strconv"
	"strings"
)

// DB maps lowercased inflected forms to their root form.
type DB struct {
	m    map[string]string
	freq map[string]int
}

// Parse builds a DB from a lemma list: lines of `root/freq -> infl1,infl2,...`,
// with `;` comments and blank lines skipped.
func Parse(r io.Reader) (*DB, error) {
	db := &DB{m: make(map[string]string), freq: make(map[string]int)}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if strings.TrimSpace(line) == "" || strings.HasPrefix(strings.TrimSpace(line), ";") {
			continue
		}
		parts := strings.Split(line, " -> ")
		if len(parts) != 2 {
			continue
		}
		rootFreq := strings.SplitN(strings.TrimSpace(parts[0]), "/", 2)
		root := strings.ToLower(strings.TrimSpace(rootFreq[0]))
		if root == "" {
			continue
		}
		freq := 0
		if len(rootFreq) == 2 {
			if n, err := strconv.Atoi(rootFreq[1]); err == nil {
				freq = n
			}
		}
		if _, exists := db.freq[root]; !exists {
			db.freq[root] = freq
		}
		if _, exists := db.m[root]; !exists {
			db.m[root] = root
		}
		for _, infl := range strings.Split(parts[1], ",") {
			infl = strings.ToLower(strings.TrimPrefix(strings.TrimSpace(infl), "!"))
			if infl == "" {
				continue
			}
			if _, exists := db.m[infl]; !exists {
				db.m[infl] = root
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return db, nil
}

var contractionSuffixes = []string{"n't", "'s", "'ll", "'d", "'ve", "'re", "'m"}

// Lookup returns the root form and frequency for a word, falling back to
// stripping a contraction suffix, and reports whether either lookup succeeded.
func (db *DB) Lookup(word string) (string, int, bool) {
	w := strings.ToLower(strings.TrimSpace(word))
	if root, ok := db.m[w]; ok {
		return root, db.freq[root], true
	}
	for _, suf := range contractionSuffixes {
		if !strings.HasSuffix(w, suf) {
			continue
		}
		if root, ok := db.m[strings.TrimSuffix(w, suf)]; ok {
			return root, db.freq[root], true
		}
	}
	return "", 0, false
}
