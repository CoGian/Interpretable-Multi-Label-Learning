wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14kAQPHHhDxqkNz3yrP07jbE69_cY80Ji' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14kAQPHHhDxqkNz3yrP07jbE69_cY80Ji" -O Datasets.zip && rm -rf /tmp/cookies.txt
unzip Datasets.zip