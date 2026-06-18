#!/opt/homebrew/bin/python3

from ansible.module_utils.basic import AnsibleModule

def walk():

    module_args = dict(
        name=dict(type='str', required=True)
    )

    module = AnsibleModule(
        argument_spec=module_args
    )

    result = {
        "changed": False,
        "message": f"Welcome, {module.params['name']}!"
    }

    module.exit_json(**result)

if __name__ == '__main__':
    walk()