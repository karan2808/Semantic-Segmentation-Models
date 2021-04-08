wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=_YOUR_USER_NAME_&password=_YOUR_PASSWORD_&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
mkdir cityscapes
unzip -q gtFine_trainvaltest.zip
unzip -q leftImg8bit_trainvaltest.zip
mv gtFine cityscapes/
mv leftImg8bit cityscapes/