#!/bin/bash
for file in $(find ./Data -name '*json'); do
	name="$(basename -s .json $file)"
	path=$(dirname "$file")
	read -ra parts <<< "$(echo $file | awk -F/ '{print $3,  $4}')"
	new_name="Data/${parts[0]}_${parts[1]}.processed"
	set -x
	jq -c '.[]' "$file" > "$new_name"
	{ set +x; } &> /dev/null
	echo "File written to $new_name"
done
