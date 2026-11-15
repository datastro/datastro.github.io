# frozen_string_literal: true

source "https://rubygems.org"

gem "jekyll-theme-chirpy", "~> 7.6"

# 이전 주소(/posts/)를 새 주소로 넘겨주는 리다이렉트
group :jekyll_plugins do
  gem "jekyll-redirect-from", "~> 0.16"
end

gem "html-proofer", "~> 5.0", group: :test

platforms :windows, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.2.0", :platforms => [:windows]
