.PHONY: default
default: data;

data: lm/en.arpa.bin lm/en.sp.model
	mkdir -p lm
	wget -c  -P lm http://dl.fbaipublicfiles.com/cc_net/lm/en.arpa.bin
	wget -c  -P lm http://dl.fbaipublicfiles.com/cc_net/lm/en.sp.model
