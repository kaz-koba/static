protected void doGet(HttpServletRequest request, HttpServletResponse resp) {
    String firstName = request.getParameter("firstName");
    resp.getWriter().append("<div>");
    resp.getWriter().append("Search for " + firstName);
    resp.getWriter().append("</div>");
}

