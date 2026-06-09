export const prerender = false;

export async function GET() {
	const vars = {
		KEYSTATIC_GITHUB_CLIENT_ID: !!import.meta.env.KEYSTATIC_GITHUB_CLIENT_ID,
		KEYSTATIC_GITHUB_CLIENT_SECRET: !!import.meta.env.KEYSTATIC_GITHUB_CLIENT_SECRET,
		KEYSTATIC_SECRET: !!import.meta.env.KEYSTATIC_SECRET,
		idPrefix: typeof import.meta.env.KEYSTATIC_GITHUB_CLIENT_ID === 'string'
			? import.meta.env.KEYSTATIC_GITHUB_CLIENT_ID.substring(0, 8)
			: 'NOT_FOUND',
	};

	return new Response(JSON.stringify(vars, null, 2), {
		headers: { 'Content-Type': 'application/json' },
	});
}
