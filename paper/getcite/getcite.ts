#!/usr/bin/env node
/// <reference path="untyped.d.ts" />
import pandoc, { FilterAction, Str, EltMap, Elt, Cite } from "pandoc-filter";
import * as fs from "fs";

import cjs from "citation-js";

const citationCachePath = "citation-cache.json";

let cache: { [url: string]: { fetched: string; bibtex: string; csl: any } } = {};

async function getCslForUrl(url: string) {
	// uses zotero extractors from https://github.com/zotero/translators to get information from URLs
	// https://www.mediawiki.org/wiki/Citoid/API
	console.error("fetching citation from url", url);
	const res = await fetch(`https://en.wikipedia.org/api/rest_v1/data/citation/bibtex/${encodeURIComponent(url)}`);

	if (!res.ok) {
		throw `Could not fetch ${url}: ${await res.text()}`;
	}
	const bibtex = await res.text();
	const cbb = new cjs(bibtex);

	if (cbb.data.length !== 1) throw Error("got != 1 response: " + bibtex);
	cbb.data[0].id = url;
	const [csl] = cbb.get({ format: "real", type: "json", style: "csl" });
	delete csl._graph;

	return { fetched: new Date().toJSON(), bibtex, csl };
}

async function getCslForUrlCached(url: string) {
	if (url in cache) return;
	cache[url] = await getCslForUrl(url);
}

async function action<A extends keyof EltMap>(
	key: A,
	value: EltMap[A],
	format: string,
	meta: any
): Promise<undefined | Elt<keyof EltMap> | Array<Elt<keyof EltMap>>> {
	if (key === "Cite") {
		const citekeys = meta.citekeys.c;
		const [citations, inline] = value as EltMap["Cite"];
		for (const citation of citations) {
			const _url = citekeys[citation.citationId];
			if (!_url) continue;
			const url = _url.c[0].c;
			await getCslForUrlCached(url);
			citation.citationId = url;
		}
	}
	return undefined;
}

function toMeta(e: string | object | (string | object)[]): any {
	if (Array.isArray(e)) {
		return { t: "MetaList", c: e.map(x => toMeta(x)) };
	}
	if (typeof e === "string" || typeof e === "number") return { t: "MetaString", c: String(e) };
	if (typeof e === "object") {
		const m: any = {};
		for (const [k, v] of Object.entries(e)) {
			m[k] = toMeta(v);
		}
		return { t: "MetaMap", c: m };
	}
	throw Error(typeof e);
}

async function go() {
	try {
		cache = JSON.parse(fs.readFileSync(citationCachePath, "utf8"));
	} catch {}
	const data = JSON.parse(fs.readFileSync(0, "utf8"));
	const format = process.argv.length > 2 ? process.argv[2] : "";
	const res = await (pandoc as any).filterAsync(data, action, format);
	console.error("got all citations");

	res.meta.references = toMeta(Object.entries(cache).map(([url, { csl }]) => csl));
	fs.writeFileSync(citationCachePath, JSON.stringify(cache, null, "\t"));
	process.stdout.write(JSON.stringify(res));
}

go();
