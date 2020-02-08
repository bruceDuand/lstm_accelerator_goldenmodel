#!/usr/bin/env perl
use strict;
use warnings;

#read and write file
open(my $weight_in, "<", "$ARGV[0]");
open(my $weight_out, ">","$ARGV[1]");
my @line = readline($weight_in);


for (my $i = 0; $i <= @line-1; $i++){
    my @singleline = split(' ', $line[$i]);
    for (my $j = 0; $j <= @singleline-1; $j++){
        my @num;
        for (my $k = 0; $k <= length($singleline[$j])-1; $k++){
            $num[$k] = substr($singleline[$j],$k,1)
        }
        print @num,"//";
        my $data = 0.0;
        if ($num[0] eq "0"){
            for (my $m=1; $m <= length($singleline[$j])-1; $m++){
                $data = $data + $num[$m] * (2 ** (0-$m));
            }
        }
        if ($num[0] eq "1"){
            for (my $m=1; $m <= length($singleline[$j])-1; $m++){
                $data = $data + $num[$m] * (2 ** (0-$m));
            }
            $data = 0 - $data;
        }
        print $data;
        print $weight_out $data," ";
        print "\n";
    }
    print "\n";
    print $weight_out "\n";
}