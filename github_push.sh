
if [ -z "$1" ]
then
  a="num 7"
else
  a="$1"
fi

git add .
git commit -m "$a"
git push -u origin main
