def align_with_obb(mesh, debug=False):
    """
    Aligns the mesh based on OBB (Oriented Bounding Box) axes.
    The shortest axis is aligned with the z-axis, the longest with the x-axis,
    and the middle-length axis with the y-axis.
    
    Detailed transformation process:
    1. Move the mesh center point (OBB center) to the origin
    2. Find principal axes using PCA to calculate rotation matrix
    3. Apply rotation matrix to align the mesh
    
    Args:
        mesh: PyVista mesh
        debug: Debug mode (visualize intermediate steps)
        
    Returns:
        aligned_mesh: Aligned mesh
        obb_center: OBB center point
        rotation_matrix: Rotation matrix
    """
    try:
        print("[Log] Starting align_with_obb function")
        # Copy mesh
        aligned_mesh = mesh.copy()
        
        # Mesh vertices
        vertices = mesh.points
        print(f"[Log] Number of vertices: {len(vertices)}")
        
        # Extract values needed for OBB calculation
        # Calculate point average (OBB center)
        mean_pt = np.mean(vertices, axis=0)
        print(f"[Log] OBB center coordinates: {mean_pt}")
        
        # Move points relative to the mean
        centered_pts = vertices - mean_pt
        print(f"[Log] Centering completed")
        
        # Calculate covariance matrix
        cov = np.cov(centered_pts, rowvar=False)
        print(f"[Log] Covariance matrix calculation completed")
        
        # Calculate eigenvalues and eigenvectors
        try:
            print(f"[Log] Starting eigenvalue calculation")
            eigvals, eigvecs = np.linalg.eigh(cov)
            print(f"[Log] Eigenvalues: {eigvals}")
            print(f"[Log] Eigenvector matrix shape: {eigvecs.shape}")
        except Exception as e:
            print(f"[Error] Error during eigenvalue calculation: {e}")
            raise
        
        # Sort eigenvalues in descending order (principal axes order)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        print(f"[Log] Sorted eigenvalues: {eigvals}")
        
        # Align axes based on eigenvalues: largest -> x-axis, middle -> y-axis, smallest -> z-axis
        rotation_matrix = eigvecs
        print(f"[Log] Rotation matrix calculation completed")
        
        # Apply rotation transformation (transform from original coordinate system to standard coordinate system)
        # Note: centered_pts are already points moved to origin
        transformed_vertices = np.dot(centered_pts, rotation_matrix)
        print(f"[Log] Vertex transformation completed")
        
        # Apply transformed vertices
        aligned_mesh.points = transformed_vertices
        print(f"[Log] Transformed mesh creation completed")
        
        # Debug mode: Visualize transformation process
        if debug:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(15, 5))
            
            # Visualize original mesh points
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='.', alpha=0.01)
            ax1.set_title('Original Mesh')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # Visualize points moved to origin
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.scatter(centered_pts[:, 0], centered_pts[:, 1], centered_pts[:, 2], c='g', marker='.', alpha=0.01)
            ax2.set_title('Centered Mesh')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            # Visualize points after rotation transformation
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.scatter(transformed_vertices[:, 0], transformed_vertices[:, 1], transformed_vertices[:, 2], c='r', marker='.', alpha=0.01)
            ax3.set_title('Rotated Mesh')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            
            plt.tight_layout()
            plt.show()
        
        print(f"[Log] align_with_obb function completed")
        return aligned_mesh, mean_pt, rotation_matrix
    except Exception as e:
        print(f"[Error] Exception occurred in align_with_obb function: {e}")
        import traceback
        traceback.print_exc()
        # Return default values even if error occurs
        return mesh.copy(), np.zeros(3), np.eye(3)