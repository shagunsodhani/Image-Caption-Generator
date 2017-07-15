cd /home/shagun/projects/Image-Caption-Generator/data/
sed -e 's/^/https:\/\/www.flickr.com\/photos\//' Flickr_8k.trainImages.txt > temp.txt
wget -i temp.txt -P /home/shagun/projects/Image-Caption-Generator/data/images
rm temp.txt

sed -e 's/^/https:\/\/www.flickr.com\/photos\//' Flickr_8k.testImages.txt > temp.txt
wget -i temp.txt -P /home/shagun/projects/Image-Caption-Generator/data/images
rm temp.txt

sed -e 's/^/https:\/\/www.flickr.com\/photos\//' Flickr_8k.devImages.txt > temp.txt
wget -i temp.txt -P /home/shagun/projects/Image-Caption-Generator/data/images
rm temp.txt

echo "success"