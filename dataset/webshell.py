import crawl

data_type = "webshell"

# 仓库列表：[(用户名, 仓库名, 描述)]
REPOS = [
    ("tennc", "webshell", ""),
    ("xl7dev", "WebShell", ""),
    ("JohnTroony", "php-webshells", ""),
    ("BlackArch", "webshells", ""),
    ("LandGrey", "webshell-detect-bypass", ""),
    ("JoyChou93", "webshell", ""),
    ("bartblaze", "PHP-backdoors", ""),
    ("WangYihang", "Webshell-Sniper", ""),
    ("threedr3am", "JSP-Webshells", ""),
    ("DeEpinGh0st", "PHP-bypass-collection", ""),
    ("lcatro", "PHP-WebShell-Bypass-WAF", ""),
    ("tanjiti", "webshellSample", ""),
    ("webshellpub", "awsome-webshell", ""),
    ("tdifg", "WebShell", ""),
    ("malwares", "WebShell", ""),
    ("lhlsec", "webshell", ""),
    ("oneoneplus", "webshell", ""),
    ("vnhacker1337", "Webshell", ""),
    ("backlion", "webshell", ""),
    ("twepl", "wso", "wso for php8"),
    ("flozz", "p0wny-shell", "p0wny-shell"),
]

crawl.git_clone(REPOS, data_type)
