#!/bin/bash
for file in $(find ./Data -name '*json'); do
	name="$(basename $file | awk -F. '{print $1}')_line_oriented.json"
	path=$(dirname "$file")
	echo "$name" "$path"
	jq -c '.[]' "$file" > "$path/$name"
done
