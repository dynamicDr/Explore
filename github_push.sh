
if [ -z "$1" ]
then
  a="(No description)"
else
  a="$1"
fi

git add D:/remote/Ensembled/.
git commit -m "$a"
git push -u origin main
