#!/bin/bash
for file in $(find ./Data -name '*json'); do
	name="$(basename -s .json $file)"
	path=$(dirname "$file")
	read -ra parts <<< "$(echo $file | awk -F/ '{print $3,  $4}')"
	new_name="Data/${parts[0]}_${parts[1]}_$name.processed"
	set -x
	jq -c '.[]' "$file" > "$new_name" # &
	{ set +x; } &> /dev/null
	echo "File will be written to $new_name"
done
# wait
