export const prerender = false;

export async function GET({ request }: { request: Request }) {
	const reqUrl = new URL(request.url);
	const callbackUrl = `${reqUrl.origin}/api/keystatic/github/oauth/callback`;

	return new Response(JSON.stringify({
		origin: reqUrl.origin,
		host: reqUrl.host,
		protocol: reqUrl.protocol,
		pathname: reqUrl.pathname,
		requestUrl: request.url,
		callbackUrl,
		headers: {
			host: request.headers.get('host'),
			forwardedProto: request.headers.get('x-forwarded-proto'),
			forwardedHost: request.headers.get('x-forwarded-host'),
		},
	}, null, 2), {
		headers: { 'Content-Type': 'application/json' },
	});
}
