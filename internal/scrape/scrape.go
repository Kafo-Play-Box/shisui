// Package scrape turns HTML into clean paragraph-separated plaintext.
//
// It parses the DOM with golang.org/x/net/html, walks it dropping chrome
// (navigation, footers, edit links, code blocks, hidden elements), and emits
// the remaining prose one paragraph per block, blocks separated by blank
// lines. This is stage 0 of the pipeline: the output feeds internal/extract.
package scrape

import (
	"io"
	"strings"

	"golang.org/x/net/html"
)

// Run reads HTML from r and writes paragraph-separated plaintext to w.
func Run(r io.Reader, w io.Writer) error {
	doc, err := html.Parse(r)
	if err != nil {
		return err
	}
	s := &runState{}
	walk(doc, s)
	out := strings.Join(s.blocks, "\n\n")
	if out == "" {
		return nil
	}
	_, err = io.Copy(w, strings.NewReader(out))
	return err
}

// runState carries the current block buffer and the emitted blocks across
// the DOM walk. Blocks are collected first so the final join with "\n\n"
// guarantees exactly one blank line between blocks and no leading or
// trailing blank lines.
type runState struct {
	buf    strings.Builder // inline text of the block being collected
	blocks []string        // finished blocks
}

// flush collapses the buffer's whitespace and emits it as one block.
// Whitespace-only buffers emit nothing.
func (s *runState) flush() {
	text := strings.Join(strings.Fields(s.buf.String()), " ")
	s.buf.Reset()
	if text != "" {
		s.blocks = append(s.blocks, text)
	}
}

// emitAlt emits an image's alt text as its own block; empty alt emits nothing.
func (s *runState) emitAlt(n *html.Node) {
	if alt := getAttr(n, "alt"); alt != "" {
		s.blocks = append(s.blocks, strings.Join(strings.Fields(alt), " "))
	}
}

// walk iterates the DOM in document order. Block elements flush the pending
// buffer, collect their subtree, and flush again on exit; structural and
// inline elements contribute text to the current buffer; skipped subtrees
// vanish entirely. The endBlock marker makes the walk iterative: it sits
// below a block's children and flushes when it surfaces.
func walk(root *html.Node, s *runState) {
	type item struct {
		n        *html.Node
		endBlock bool
	}
	stack := []item{{n: root}}
	for len(stack) > 0 {
		it := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if it.endBlock {
			s.flush()
			continue
		}
		n := it.n
		switch n.Type {
		case html.TextNode:
			s.buf.WriteString(n.Data)
			continue
		case html.CommentNode, html.DoctypeNode:
			continue
		}
		if n.Type != html.ElementNode && n.Type != html.DocumentNode {
			continue
		}
		if shouldSkip(n) {
			continue
		}
		switch n.Data {
		case "br":
			s.buf.WriteByte(' ')
			continue
		case "img":
			s.emitAlt(n)
			continue
		case "meta", "link":
			continue
		}
		if isBlock(n) {
			s.flush()
			stack = append(stack, item{endBlock: true})
			for c := n.LastChild; c != nil; c = c.PrevSibling {
				stack = append(stack, item{n: c})
			}
			continue
		}
		for c := n.LastChild; c != nil; c = c.PrevSibling {
			stack = append(stack, item{n: c})
		}
	}
	s.flush()
}

// shouldSkip reports whether a subtree must be dropped entirely: chrome,
// code, citations, and hidden elements. Checked when entering each element.
// html and body are structural containers that carry the skin's feature-flag
// classes (vector-toc-available, vector-sticky-header-enabled, ...); they
// must never be dropped via class/id matching.
func shouldSkip(n *html.Node) bool {
	switch n.Data {
	case "html", "body":
		return false
	case "script", "style", "noscript", "nav", "header", "footer", "aside", "form", "pre":
		return true
	case "sup":
		return hasClassToken(n, "reference")
	}
	if getAttr(n, "role") == "navigation" {
		return true
	}
	if hasAttr(n, "hidden") || getAttr(n, "aria-hidden") == "true" {
		return true
	}
	if st := getAttr(n, "style"); strings.Contains(st, "display:none") || strings.Contains(st, "display: none") {
		return true
	}
	if id := getAttr(n, "id"); dropIDs[id] {
		return true
	}
	return hasDropClass(n)
}

// dropIDs are exact id attribute values of chrome subtrees.
var dropIDs = map[string]bool{
	"mw-navigation": true, "catlinks": true, "footer": true, "mw-panel-toc": true,
	"vector-toc": true, "siteSub": true, "contentSub": true, "mw-head": true,
	"p-navigation": true, "p-tb": true, "p-lang": true,
}

// dropClassContains are class tokens that mark chrome when they appear as a
// substring of a token; "toc" is matched as a whole token only so that
// vector-toc-style tokens (already covered above) and plain "toc" both drop
// without nuking unrelated words.
var dropClassContains = []string{
	"vector-toc", "mw-editsection", "printfooter", "catlinks", "mw-jump-link",
	"mw-indicators", "siteSub", "contentSub", "vector-menu", "vector-page-tools",
	"vector-appearance", "vector-sticky-header", "mw-normal-catlinks",
}

func hasDropClass(n *html.Node) bool {
	for _, tok := range strings.Fields(getAttr(n, "class")) {
		if tok == "toc" {
			return true
		}
		for _, s := range dropClassContains {
			if strings.Contains(tok, s) {
				return true
			}
		}
	}
	return false
}

// isBlock reports whether an element emits its collected text as one block.
func isBlock(n *html.Node) bool {
	switch n.Data {
	case "p", "li", "dt", "dd", "blockquote", "figcaption", "td", "th", "caption":
		return true
	}
	if len(n.Data) == 2 && n.Data[0] == 'h' && n.Data[1] >= '1' && n.Data[1] <= '6' {
		return true
	}
	return n.Data == "div" && hasClassToken(n, "archwiki-template-box")
}

func hasClassToken(n *html.Node, substr string) bool {
	for _, tok := range strings.Fields(getAttr(n, "class")) {
		if strings.Contains(tok, substr) {
			return true
		}
	}
	return false
}

func getAttr(n *html.Node, name string) string {
	for _, a := range n.Attr {
		if a.Key == name {
			return a.Val
		}
	}
	return ""
}

func hasAttr(n *html.Node, name string) bool {
	for _, a := range n.Attr {
		if a.Key == name {
			return true
		}
	}
	return false
}
