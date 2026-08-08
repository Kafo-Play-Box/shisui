.PHONY: build fmt vet staticcheck test check clean

build:
	go build -trimpath -o shisui .

fmt:
	@test -z "$$(gofmt -l .)"

vet:
	go vet ./...

staticcheck:
	go run honnef.co/go/tools/cmd/staticcheck@v0.7.0 ./...

test:
	go test ./... -count=1

check: fmt vet staticcheck test

clean:
	rm -f shisui
