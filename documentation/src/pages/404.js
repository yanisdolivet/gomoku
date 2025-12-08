import React from 'react';
import Layout from '@theme/Layout';

export default function NotFound() {
    return (
        <Layout
            title="404 - Subject Not Found"
            description="The requested test chamber could not be located.">
            <main className="container margin-vert--xl">
                <div className="row">
                    <div className="col col--6 col--offset-3">
                        <h1 className="hero__title">404 - Test Chamber Not Found</h1>
                        <p>
                            The requested file has been incinerated or never existed.
                        </p>
                        <p>
                            Please return to the <a href="/">Testing Area</a> immediately.
                        </p>
                        <div style={{
                            padding: '2rem',
                            backgroundColor: '#f5f6f7',
                            borderRadius: '8px',
                            textAlign: 'center',
                            marginTop: '2rem'
                        }}>
                            <h3>Aperture Science Note:</h3>
                            <p>If you are looking for the cake, it is not here.</p>
                        </div>
                    </div>
                </div>
            </main>
        </Layout>
    );
}
