// Package jsonl reads and writes newline-delimited JSON.
package jsonl

import (
	"encoding/json"
	"io"
)

// Decoder reads one JSON value per line.
type Decoder[T any] struct {
	dec *json.Decoder
}

// NewDecoder returns a Decoder reading from r.
func NewDecoder[T any](r io.Reader) *Decoder[T] {
	return &Decoder[T]{dec: json.NewDecoder(r)}
}

// Decode reads the next value, returning io.EOF at the end of input.
func (d *Decoder[T]) Decode() (T, error) {
	var v T
	err := d.dec.Decode(&v)
	return v, err
}

// Encoder writes one JSON value per line.
type Encoder[T any] struct {
	enc *json.Encoder
}

// NewEncoder returns an Encoder writing to w with HTML escaping disabled.
func NewEncoder[T any](w io.Writer) *Encoder[T] {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	return &Encoder[T]{enc: enc}
}

// Encode writes v followed by a newline.
func (e *Encoder[T]) Encode(v T) error {
	return e.enc.Encode(v)
}

// ReadAll reads every row in r.
func ReadAll[T any](r io.Reader) ([]T, error) {
	var rows []T
	d := NewDecoder[T](r)
	for {
		v, err := d.Decode()
		if err == io.EOF {
			return rows, nil
		}
		if err != nil {
			return rows, err
		}
		rows = append(rows, v)
	}
}

// WriteAll writes every row in rows to w.
func WriteAll[T any](w io.Writer, rows []T) error {
	e := NewEncoder[T](w)
	for _, row := range rows {
		if err := e.Encode(row); err != nil {
			return err
		}
	}
	return nil
}
