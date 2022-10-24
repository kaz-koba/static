import { NextPageContext } from 'next';
import Error, { ErrorProps } from 'next/error';

export default function ErrorPage(props: ErrorProps): JSX.Element {
  return <Error {...props} />;
}

// add getInitialProps to disable "https://nextjs.org/docs/messages/prerender-error"
//   Error: Export encountered errors on following paths:
//     /_error: /404
//     /_error: /500
// see: https://github.com/vercel/next.js/issues/23568#issuecomment-814971318
ErrorPage.getInitialProps = (ctx: NextPageContext) => {
  const { res, err } = ctx;
  const statusCode = res?.statusCode ?? err?.statusCode ?? 500;
  return { statusCode };
};
