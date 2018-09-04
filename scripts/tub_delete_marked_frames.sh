#!/bin/bash


if [ "x${1}" == 'xCHECK_DELETE' ]; then
	echo -n $2...
	JPEG_PATH=`echo $2 | sed "s/record_//g" | sed "s/.json/_cam-image_array_.jpg/g"`
	grep -q delete $2 && (echo DELETE; rm -vf $2 $JPEG_PATH) || echo KEEP
	exit
fi

if [ "x${1}" == "x" ]; then
	echo "Deletes frames marked for deletion in json file."
	echo USAGE: $0 FOLDER_WITH_RECORD_JSONS
	exit
fi

echo Trying to delete tub frames marked in json for deletion.
find $1 -name 'record_*.json' -exec $0 CHECK_DELETE {} \;
