// Load configuration from environment or config file
const path = require('path');
const webpack = require('webpack');

module.exports = {
  webpack: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
    configure: (webpackConfig) => {
      const hasHmrPlugin = webpackConfig.plugins.some(
        (plugin) => plugin && plugin.constructor && plugin.constructor.name === 'HotModuleReplacementPlugin'
      );

      if (webpackConfig.mode === 'development' && !hasHmrPlugin) {
        webpackConfig.plugins.push(new webpack.HotModuleReplacementPlugin());
      }

      webpackConfig.watchOptions = {
        ...webpackConfig.watchOptions,
        ignored: [
          '**/node_modules/**',
          '**/.git/**',
          '**/build/**',
          '**/dist/**',
          '**/coverage/**',
          '**/public/**',
        ],
      };

      return webpackConfig;
    },
  },
};
