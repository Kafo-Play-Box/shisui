package jsonl

import (
	"bytes"
	"io"
	"reflect"
	"strings"
	"testing"
)

type testRow struct {
	Name string   `json:"name"`
	N    int      `json:"n"`
	Tags []string `json:"tags,omitempty"`
}

func TestRoundTrip(t *testing.T) {
	rows := []testRow{
		{Name: "a&b", N: 1, Tags: []string{"x", "y"}},
		{Name: "c", N: 2},
	}
	var buf bytes.Buffer
	if err := WriteAll(&buf, rows); err != nil {
		t.Fatal(err)
	}
	got, err := ReadAll[testRow](&buf)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(rows) {
		t.Fatalf("read %d rows, want %d", len(got), len(rows))
	}
	for i := range rows {
		if !reflect.DeepEqual(got[i], rows[i]) {
			t.Errorf("row %d = %+v, want %+v", i, got[i], rows[i])
		}
	}
}

func TestEncoderDisablesHTMLEscape(t *testing.T) {
	var buf bytes.Buffer
	e := NewEncoder[testRow](&buf)
	if err := e.Encode(testRow{Name: "<b>&</b>", N: 0}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "<b>&</b>") {
		t.Errorf("output %q should keep raw HTML", buf.String())
	}
}

func TestEmptyInput(t *testing.T) {
	rows, err := ReadAll[testRow](strings.NewReader(""))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 0 {
		t.Fatalf("read %d rows, want 0", len(rows))
	}
}

func TestMalformedLine(t *testing.T) {
	if _, err := ReadAll[testRow](strings.NewReader("not json\n")); err == nil {
		t.Fatal("expected error for malformed line")
	}
}

func TestDecoderEOF(t *testing.T) {
	d := NewDecoder[testRow](strings.NewReader(`{"name":"a","n":1}` + "\n"))
	if _, err := d.Decode(); err != nil {
		t.Fatal(err)
	}
	if _, err := d.Decode(); err != io.EOF {
		t.Fatalf("second Decode = %v, want io.EOF", err)
	}
}
