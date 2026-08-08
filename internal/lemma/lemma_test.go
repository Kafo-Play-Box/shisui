package lemma

import (
	"strings"
	"testing"
)

const testData = `; comment line
be/4109826 -> is,was,are,'s,been,being
run/44715 -> running,ran,runs
cat/5400 -> cats
ally/2831 -> allies,allied,allying,!allies
do/535646 -> did,does,done,doing
it/1213224 -> its
'hood -> 'hoods
`

func mustParse(t *testing.T) *DB {
	t.Helper()
	db, err := Parse(strings.NewReader(testData))
	if err != nil {
		t.Fatal(err)
	}
	return db
}

func TestLookup(t *testing.T) {
	db := mustParse(t)
	cases := []struct {
		word string
		root string
		freq int
		ok   bool
	}{
		{"running", "run", 44715, true},
		{"ran", "run", 44715, true},
		{"was", "be", 4109826, true},
		{"be", "be", 4109826, true},
		{"cats", "cat", 5400, true},
		{"allies", "ally", 2831, true},
		{"allied", "ally", 2831, true},
		{"don't", "do", 535646, true},
		{"it's", "it", 1213224, true},
		{"Running", "run", 44715, true},
		{"'hoods", "'hood", 0, true},
		{"nonexistent", "", 0, false},
		{"", "", 0, false},
	}
	for _, c := range cases {
		root, freq, ok := db.Lookup(c.word)
		if ok != c.ok || root != c.root || freq != c.freq {
			t.Errorf("Lookup(%q) = (%q, %d, %v), want (%q, %d, %v)", c.word, root, freq, ok, c.root, c.freq, c.ok)
		}
	}
}

func TestParseSkipsCommentsAndBlankLines(t *testing.T) {
	db := mustParse(t)
	if root, freq, ok := db.Lookup("is"); !ok || root != "be" || freq != 4109826 {
		t.Errorf("Lookup(is) = (%q, %d, %v), want (be, 4109826, true)", root, freq, ok)
	}
	if root, freq, ok := db.Lookup("'hoods"); !ok || root != "'hood" || freq != 0 {
		t.Errorf("Lookup('hoods) = (%q, %d, %v), want ('hood, 0, true)", root, freq, ok)
	}
	if root, ok := db.m["cats"]; !ok || root != "cat" {
		t.Errorf("db.m[cats] = (%q, %v), want (cat, true)", root, ok)
	}
}

func TestFreqFirstWriteWins(t *testing.T) {
	db, err := Parse(strings.NewReader("root/100 -> roots\nroot/999 -> roots\n"))
	if err != nil {
		t.Fatal(err)
	}
	if root, freq, ok := db.Lookup("roots"); !ok || root != "root" || freq != 100 {
		t.Errorf("Lookup(roots) = (%q, %d, %v), want (root, 100, true)", root, freq, ok)
	}
}
