package main

import _ "embed"

// EmbeddedLemma is the bundled lemma database.
//
//go:embed lemma.en.txt
var EmbeddedLemma []byte
