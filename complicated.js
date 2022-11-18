const doGet = (request, resp) => {
  let firstName = request.getParameter("firstName");
  resp.getWriter().append("<div>");
  resp.getWriter().append("Search for " + firstName);
  resp.getWriter().append("</div>");
};
