// @ts-check

import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import keystatic from '@keystatic/astro';
import react from '@astrojs/react';
import vercel from '@astrojs/vercel';
import { defineConfig, fontProviders } from 'astro/config';

// https://astro.build/config
export default defineConfig({
	site: 'https://my-blog-lake-one-53.vercel.app',
	adapter: vercel(),
	integrations: [
		mdx(),
		sitemap(),
		react(),
		keystatic({
			clientId: process.env.KEYSTATIC_GITHUB_CLIENT_ID,
			clientSecret: process.env.KEYSTATIC_GITHUB_CLIENT_SECRET,
			secret: process.env.KEYSTATIC_SECRET,
		}),
	],
	fonts: [
		{
			provider: fontProviders.local(),
			name: 'Atkinson',
			cssVariable: '--font-atkinson',
			fallbacks: ['sans-serif'],
			options: {
				variants: [
					{
						src: ['./src/assets/fonts/atkinson-regular.woff'],
						weight: 400,
						style: 'normal',
						display: 'swap',
					},
					{
						src: ['./src/assets/fonts/atkinson-bold.woff'],
						weight: 700,
						style: 'normal',
						display: 'swap',
					},
				],
			},
		},
	],
});
