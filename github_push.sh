
if [ -z "$1" ]
then
  a="Pass num 1"
else
  a="$1"
fi

git add .
git commit -m "$a"
git push -u origin main
