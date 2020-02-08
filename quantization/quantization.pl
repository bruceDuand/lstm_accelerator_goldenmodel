#!/usr/bin/env perl
use strict;
use warnings;

#read and write file
open(my $weight_in, "<", "$ARGV[0]");
open(my $weight_out, ">","$ARGV[1]");
my @line = readline($weight_in);

my $text;
my $num_parameter = 0;

for (my $i = 1; $i <= @line-1; $i++){
    my @singleline = split(' ', $line[$i]);
    if ($singleline[0] eq "#"){
        next;
    }
    
    for (my $j = 0; $j <= @singleline-1; $j++){
        my @num;
        my @num_out;
        for (my $k = 0; $k <= length($singleline[$j]); $k++){
            $num[$k] = substr($singleline[$j],$k,1); 
        }
        my $data=0.0;
        print @num,"//";
        $num_parameter = $num_parameter + 1;
        if ($num[0] eq "-"){
            $num_out[0] = 1;
            for (my $m = 3; $m <= length($singleline[$j])-1; $m++){
                $data = $data + $num[$m] * 10 ** (0-($m-2));
            }
        }
        else {
            $num_out[0] = 0; 
            for (my $m = 2; $m <= length($singleline[$j])-1; $m++){
                $data = $data + $num[$m] * 10 ** (0-($m-1));
            }
        }
        my $bidata;
        for (my $n = 1; $n <= 7; $n++){
            $bidata = int($data * 2);
            $data = $data * 2;
            $data = $data - $bidata;
            $num_out[$n] = $bidata;
        }
        print @num_out;
        print $weight_out @num_out," ";
        print "\n";
    
    }
    print $weight_out "\n";
}

print $num_parameter,"\n";
