import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from '../css/index.module.css';

function HomepageHeader() {
    const { siteConfig } = useDocusaurusContext();
    return (
        <header className={clsx('hero hero--primary', styles.heroBanner)} style={{ backgroundColor: '#1b1b1d', backgroundImage: 'linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5))', backgroundSize: 'cover', backgroundPosition: 'center' }}>
            <div className="container">
                <h1 className="hero__title" style={{ fontSize: '4rem', fontWeight: 'bold' }}>{siteConfig.title}</h1>
                <p className="hero__subtitle" style={{ fontSize: '1.5rem' }}>{siteConfig.tagline}</p>
                <div className={styles.buttons}>
                    <Link
                        className="button button--secondary button--lg"
                        to="/docs/Introduction">
                        Get Started 🚀
                    </Link>
                </div>
            </div>
        </header>
    );
}

function Feature({ title, description }) {
    return (
        <div className={clsx('col col--4')}>
            <div className={clsx('text--center', styles.feature)}>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
        </div>
    );
}

export default function Home() {
    const { siteConfig } = useDocusaurusContext();
    return (
        <Layout
            title={`Welcome to ${siteConfig.title}`}
            description="Gomoku AI Documentation">
            <HomepageHeader />
            <main>
                <section style={{ padding: '4rem 0' }}>
                    <div className="container">
                        <div className="row">
                            <Feature
                                title="Advanced Strategy"
                                description="Implements Minimax algorithm with Alpha-Beta pruning for optimal move decision making."
                            />
                            <Feature
                                title="High Performance"
                                description="Written in C++ for maximum execution speed, crucial for real-time game AI."
                            />
                            <Feature
                                title="Pbrain Protocol"
                                description="Fully compatible with the Pbrain protocol, ready for competitions and integration with Gomoku interfaces."
                            />
                        </div>
                    </div>
                </section>
                <div style={{ textAlign: 'center', padding: '2rem', backgroundColor: '#f5f6f7', color: '#1b1b1d' }}>
                    <h2>Ready to challenge the AI?</h2>
                    <p>Check out the installation guide to get started.</p>
                </div>
            </main>
        </Layout>
    );
}