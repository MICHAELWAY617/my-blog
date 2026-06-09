import { config, collection, fields } from '@keystatic/core';

export default config({
	storage: {
		kind: 'github',
		repo: {
			owner: 'MICHAELWAY617',
			name: 'my-blog',
		},
	},
	ui: {
		brand: { name: 'WayLog' },
	},
	collections: {
		blog: collection({
			label: '博客文章',
			path: 'src/content/blog/*',
			slugField: 'title',
			format: {
				contentField: 'content',
			},
			schema: {
				title: fields.slug({
					name: {
						label: '标题',
						validation: { isRequired: true },
					},
				}),
				pubDate: fields.date({
					label: '发布日期',
					defaultValue: { kind: 'today' },
				}),
				updatedDate: fields.date({
					label: '更新日期',
				}),
				description: fields.text({
					label: '摘要',
					validation: { isRequired: true },
					multiline: true,
				}),
				heroImage: fields.image({
					label: '封面图',
					directory: 'src/assets',
					publicPath: '../../assets',
				}),
				content: fields.markdoc({
					label: '正文',
					extension: 'md',
				}),
			},
		}),
	},
});
