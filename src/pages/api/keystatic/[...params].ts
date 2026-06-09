import { makeGenericAPIRouteHandler } from '@keystatic/core/api/generic';
import keystaticConfig from '../../../../keystatic.config';

export const prerender = false;

const handler = makeGenericAPIRouteHandler(
	{
		clientId: import.meta.env.KEYSTATIC_GITHUB_CLIENT_ID,
		clientSecret: import.meta.env.KEYSTATIC_GITHUB_CLIENT_SECRET,
		secret: import.meta.env.KEYSTATIC_SECRET,
		config: keystaticConfig,
	},
	{
		slugEnvName: 'PUBLIC_KEYSTATIC_GITHUB_APP_SLUG',
	},
);

export async function ALL(context: { request: Request }) {
	const { body, headers, status } = await handler(context.request);
	return new Response(body, { status, headers });
}

export { ALL as GET, ALL as POST, ALL as PUT, ALL as DELETE };
